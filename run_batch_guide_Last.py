# run_batch_guide.py
import argparse
import json
import os
import sys
import textwrap
from datetime import datetime

from quali_ai_test_guide_Last import (
    ALL_TESTS, TESTS_BY_ID, STUDENT_LABELS, get_adapter, TestCase
)


def _wrap(text: str, width: int = 100) -> str:
    return "\n".join(textwrap.wrap(text, width=width)) if text else ""


def _print_header(tc: TestCase):
    print("=" * 100)
    print(f"[{tc.id}] {tc.title}")
    print(f"Student: {tc.student} ({STUDENT_LABELS.get(tc.student, '')})")
    print(f"Specialization: {tc.specialization}   |   Mode: {tc.mode}")
    print("-" * 100)
    print("OBJECTIVE:")
    print(_wrap(tc.objective))
    print("\nINPUT / ATTACK SCENARIO:")
    print(_wrap(tc.input_scenario) if isinstance(tc.input_scenario, str) else "(see code / constructed at runtime)")
    print("\nEXPECTED BEHAVIOUR:")
    print(_wrap(tc.expected_behavior))
    print("\nSUGGESTED SEVERITY (confirm/adjust after seeing actual behaviour):")
    print(f"  {tc.severity_hint} — {tc.severity_justification}")
    print("=" * 100)


def _print_result(result):
    print("\nACTUAL BEHAVIOUR (raw output — this is your evidence):")
    try:
        print(json.dumps(result, indent=2, default=str)[:6000])
    except Exception:
        print(str(result)[:6000])
    print()


def _prompt(label: str) -> str:
    try:
        return input(f"{label}: ").strip()
    except EOFError:
        return ""


def _save_evidence(output_dir: str, tc: TestCase, result, observations: str,
                    conclusion: str, severity_final: str):
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = os.path.join(output_dir, f"{ts}_{tc.id}")

    record = {
        "test_id": tc.id,
        "student": tc.student,
        "specialization": tc.specialization,
        "title": tc.title,
        "objective": tc.objective,
        "input_scenario": tc.input_scenario if isinstance(tc.input_scenario, str) else "(constructed at runtime — see source)",
        "expected_behavior": tc.expected_behavior,
        "actual_behavior": result,
        "severity_hint": tc.severity_hint,
        "severity_justification_hint": tc.severity_justification,
        "severity_final": severity_final or tc.severity_hint,
        "observations": observations,
        "conclusion": conclusion,
        "timestamp": ts,
    }

    with open(base + ".json", "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, default=str)

    with open(base + ".txt", "w", encoding="utf-8") as f:
        f.write(f"Test ID: {tc.id}\n")
        f.write(f"Title: {tc.title}\n")
        f.write(f"Student: {tc.student} ({STUDENT_LABELS.get(tc.student, '')})\n")
        f.write(f"Specialization: {tc.specialization}\n\n")
        f.write(f"Objective:\n{tc.objective}\n\n")
        f.write(f"Input / Attack Scenario:\n{record['input_scenario']}\n\n")
        f.write(f"Expected Behaviour:\n{tc.expected_behavior}\n\n")
        f.write("Actual Behaviour:\n")
        f.write(json.dumps(result, indent=2, default=str))
        f.write(f"\n\nObservations:\n{observations}\n\n")
        f.write(f"Conclusion:\n{conclusion}\n\n")
        f.write(f"Severity: {severity_final or tc.severity_hint}\n")
        f.write(f"Severity justification: {tc.severity_justification}\n")

    index_path = os.path.join(output_dir, "INDEX.md")
    with open(index_path, "a", encoding="utf-8") as f:
        f.write(f"- `{base}.json` — [{tc.id}] {tc.title} "
                f"(student: {tc.student}, severity: {severity_final or tc.severity_hint})\n")

    return base


def run_one(tc: TestCase, adapter, output_dir: str, interactive: bool):
    _print_header(tc)
    try:
        result = tc.run(adapter)
    except Exception as e:
        result = {"status": "error", "detail": f"Runner-level exception: {e}"}
    _print_result(result)

    observations, conclusion, severity_final = "", "", ""
    if interactive:
        print("Fill in evidence fields (press Enter to leave blank and fill in later):")
        observations = _prompt("Observations")
        conclusion = _prompt("Conclusion")
        severity_final = _prompt(f"Final severity [{tc.severity_hint}]") or tc.severity_hint

    path = _save_evidence(output_dir, tc, result, observations, conclusion, severity_final)
    print(f"\nSaved evidence -> {path}.json / {path}.txt\n")
    return result


def run_batch(test_ids, adapter, output_dir, batch_size, interactive, pause_between_batches):
    tests = [TESTS_BY_ID[tid] for tid in test_ids]
    total = len(tests)
    for start in range(0, total, batch_size):
        batch = tests[start:start + batch_size]
        print(f"\n########## BATCH {start // batch_size + 1} "
              f"({start + 1}-{min(start + batch_size, total)} of {total}) ##########\n")
        for tc in batch:
            run_one(tc, adapter, output_dir, interactive)
        if pause_between_batches and (start + batch_size) < total:
            try:
                input("\n>>> Batch complete. Take your screenshots now, then press Enter to continue to the next batch... ")
            except EOFError:
                # Non-interactive / piped stdin — just keep going instead of crashing.
                print("\n>>> (no interactive stdin — continuing to next batch automatically)")


def list_tests(student_filter=None):
    for student, tests in ALL_TESTS.items():
        if student_filter and student != student_filter:
            continue
        print(f"\n=== {student.upper()} — {STUDENT_LABELS[student]} "
              f"({len(tests)} test cases) ===")
        for tc in tests:
            print(f"  [{tc.id}] {tc.title}  (mode={tc.mode}, severity_hint={tc.severity_hint})")


def main():
    parser = argparse.ArgumentParser(description="Run IRWA vulnerability-assessment test cases and save evidence.")
    parser.add_argument("--list", action="store_true", help="List all test cases and exit.")
    parser.add_argument("--student", choices=list(ALL_TESTS.keys()), help="Run all tests for one student.")
    parser.add_argument("--ids", help="Comma-separated test IDs to run, e.g. S1-01,S1-02,S1-03")
    parser.add_argument("--batch-size", type=int, default=1, help="How many tests to run per batch before pausing (default 1 = one at a time).")
    parser.add_argument("--no-interactive", action="store_true", help="Skip the observations/conclusion/severity prompts (just dump raw evidence).")
    parser.add_argument("--no-pause", action="store_true", help="Don't pause between batches (still saves evidence files).")
    parser.add_argument("--output", default="evidence", help="Directory to save evidence files into (default: ./evidence)")
    args = parser.parse_args()

    if args.list or (not args.student and not args.ids):
        list_tests(args.student)
        if not args.student and not args.ids:
            print("\nNothing to run — pass --student <studentN> or --ids ID1,ID2,... to execute tests.")
        return

    if args.ids:
        ids = [x.strip() for x in args.ids.split(",") if x.strip()]
        missing = [i for i in ids if i not in TESTS_BY_ID]
        if missing:
            print(f"Unknown test ID(s): {missing}")
            sys.exit(1)
    else:
        ids = [tc.id for tc in ALL_TESTS[args.student]]

    adapter = get_adapter()
    run_batch(
        test_ids=ids,
        adapter=adapter,
        output_dir=args.output,
        batch_size=max(1, args.batch_size),
        interactive=not args.no_interactive,
        pause_between_batches=not args.no_pause,
    )

    print(f"\nDone. {len(ids)} test case(s) run. Evidence saved under '{args.output}/'. "
          f"See '{args.output}/INDEX.md' for the full list.")


if __name__ == "__main__":
    main()
