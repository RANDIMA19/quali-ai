"""
Agent 4: Synthesis - Root-Cause Diagnosis
Combines entity extraction (Agent 1), anomaly detection (Agent 2), and
historical incident retrieval (Agent 3) to generate an AI-assisted diagnostic
recommendation using Ollama (local, free).
"""

import os
import json
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Ollama configuration
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434/api/generate")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "mistral")


def call_ollama(prompt: str, model_name: str, timeout: int = 600) -> str:
    """
    Call Ollama with STREAMING enabled.

    Why streaming: with stream=False, Ollama sends nothing until the whole
    answer is generated, so a slow CPU run trips the read-timeout. With
    stream=True, tokens arrive continuously and the timeout clock keeps
    resetting, so it never trips. We also force JSON output and cap the
    length so it finishes fast.
    """
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": True,          # <-- key fix: continuous tokens, no timeout wall
        "format": "json",        # <-- forces Ollama to emit valid JSON only
        "options": {
            "temperature": 0.3,  # lower = more consistent structured output
            "num_predict": 800   # cap length; 2048 was slow to finish on CPU
        }
    }

    resp = requests.post(OLLAMA_URL, json=payload, stream=True, timeout=timeout)
    resp.raise_for_status()

    parts = []
    for line in resp.iter_lines():
        if not line:
            continue
        chunk = json.loads(line)
        parts.append(chunk.get("response", ""))
        if chunk.get("done"):
            break
    return "".join(parts)


def synthesize_diagnosis(
    entities: dict,
    anomaly_findings: dict,
    historical_incidents: list,
    model: str = None
) -> dict:
    """
    Synthesize a root-cause diagnosis by combining inputs from all three agents.

    Args:
        entities: Extracted entities from Agent 1 (loom_id, batch_id, metric, defect_symptom)
        anomaly_findings: Anomaly detection results from Agent 2
        historical_incidents: Retrieved similar incidents from Agent 3
        model: Model name (optional, uses default from OLLAMA_MODEL env var)

    Returns:
        JSON dict containing the AI-assisted diagnostic recommendation.
    """

    # Build the prompt for Ollama
    prompt = f"""You are an expert textile quality assurance analyst with deep knowledge of
loom operations, batch processing, and defect diagnosis. Your task is to synthesize information
from multiple sources to provide a structured root-cause diagnosis for production incidents.

You must:
1. Analyze the extracted entities to understand the current incident context
2. Review anomaly detection findings to identify statistical outliers
3. Examine historical incidents with similar patterns to identify recurring issues
4. Provide a structured diagnosis that cites specific incident_ids from historical data
5. Offer actionable recommendations based on the analysis

Your response must be in valid JSON format with the following structure:
{{
    "root_cause_analysis": {{
        "primary_cause": "Brief statement of the most likely root cause",
        "contributing_factors": ["List of contributing factors"],
        "severity": "low|medium|high|critical"
    }},
    "cited_incidents": [
        {{
            "incident_id": "INC-XXXX",
            "relevance": "Explanation of why this incident is relevant",
            "similarity_score": 0.95
        }}
    ],
    "recommendations": [
        {{
            "action": "Specific action to take",
            "priority": "immediate|short-term|long-term",
            "expected_outcome": "Expected result"
        }}
    ],
    "confidence": "low|medium|high",
    "additional_notes": "Any other relevant observations"
}}

Be specific and evidence-based. Always cite historical incident_ids when referencing past incidents.

=== EXTRACTED ENTITIES (Agent 1) ===
{json.dumps(entities, indent=2)}

=== ANOMALY FINDINGS (Agent 2) ===
{json.dumps(anomaly_findings, indent=2)}

=== HISTORICAL INCIDENTS (Agent 3) ===
{json.dumps(historical_incidents, indent=2)}

Based on this information, provide a structured root-cause diagnosis that cites specific
incident_ids from the historical incidents when relevant. Respond with valid JSON only, no other text."""

    model_name = model or OLLAMA_MODEL

    input_summary = {
        "entities": entities,
        "anomaly_count": anomaly_findings.get("anomaly_count", 0) if anomaly_findings else 0,
        "historical_incidents_reviewed": len(historical_incidents)
    }

    try:
        response_text = call_ollama(prompt, model_name)

        # Extract JSON from the response (robust even if extra text sneaks in)
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            diagnosis_data = json.loads(response_text[json_start:json_end])
        else:
            diagnosis_data = json.loads(response_text)

        return {
            "label": "AI-Assisted Diagnostic Recommendation",
            "diagnosis": diagnosis_data,
            "input_summary": input_summary
        }

    except json.JSONDecodeError:
        return {
            "label": "AI-Assisted Diagnostic Recommendation",
            "error": "Failed to parse AI response as JSON",
            "raw_response": response_text,
            "input_summary": input_summary
        }

    except Exception as e:
        return {
            "label": "AI-Assisted Diagnostic Recommendation",
            "error": str(e),
            "input_summary": input_summary
        }


def format_diagnosis(diagnosis_result: dict) -> str:
    """Format the diagnosis result as a human-readable string."""
    output = []
    output.append("=" * 80)
    output.append(diagnosis_result["label"])
    output.append("=" * 80)

    if "error" in diagnosis_result:
        output.append(f"ERROR: {diagnosis_result['error']}")
        if "raw_response" in diagnosis_result:
            output.append("\nRaw Response:")
            output.append(diagnosis_result["raw_response"])
        return "\n".join(output)

    diag = diagnosis_result["diagnosis"]

    if "root_cause_analysis" in diag:
        rca = diag["root_cause_analysis"]
        output.append("\nROOT CAUSE ANALYSIS")
        output.append("-" * 40)
        output.append(f"Primary Cause: {rca.get('primary_cause', 'N/A')}")
        output.append(f"Severity: {rca.get('severity', 'N/A')}")
        if rca.get("contributing_factors"):
            output.append("\nContributing Factors:")
            for factor in rca["contributing_factors"]:
                output.append(f"  - {factor}")

    if "cited_incidents" in diag:
        output.append("\nCITED HISTORICAL INCIDENTS")
        output.append("-" * 40)
        for citation in diag["cited_incidents"]:
            output.append(f"\nIncident ID: {citation.get('incident_id', 'N/A')}")
            output.append(f"Relevance: {citation.get('relevance', 'N/A')}")
            output.append(f"Similarity Score: {citation.get('similarity_score', 'N/A')}")

    if "recommendations" in diag:
        output.append("\nRECOMMENDATIONS")
        output.append("-" * 40)
        for rec in diag["recommendations"]:
            output.append(f"\nAction: {rec.get('action', 'N/A')}")
            output.append(f"Priority: {rec.get('priority', 'N/A')}")
            output.append(f"Expected Outcome: {rec.get('expected_outcome', 'N/A')}")

    output.append(f"\nConfidence Level: {diag.get('confidence', 'N/A')}")

    if diag.get("additional_notes"):
        output.append(f"\nAdditional Notes: {diag['additional_notes']}")

    summary = diagnosis_result["input_summary"]
    output.append("\n" + "=" * 80)
    output.append("INPUT SUMMARY")
    output.append("-" * 40)
    output.append(f"Entities: {json.dumps(summary['entities'], indent=2)}")
    output.append(f"Anomalies Detected: {summary['anomaly_count']}")
    output.append(f"Historical Incidents Reviewed: {summary['historical_incidents_reviewed']}")

    return "\n".join(output)


# ── Example usage ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    sample_entities = {
        "loom_id": "12",
        "batch_id": "B102",
        "metric": "elongation",
        "defect_symptom": "inconsistent"
    }

    sample_anomalies = {
        "metric": "elongation",
        "total_readings": 100,
        "anomaly_count": 5,
        "anomaly_percentage": 5.0,
        "anomalies": [
            {"index": 10, "value": 150.0, "z_score": 3.5},
            {"index": 20, "value": 40.0, "z_score": -4.2}
        ],
        "statistics": {"mean": 100.0, "std": 10.0, "min": 40.0, "max": 150.0}
    }

    sample_incidents = [
        {
            "incident_id": "INC-0042",
            "reason": "inconsistent elongation on loom 12",
            "action_plan": "adjusted tension settings and recalibrated sensors",
            "similarity_score": 0.92
        },
        {
            "incident_id": "INC-0156",
            "reason": "high elongation variance in batch B102",
            "action_plan": "replaced worn rollers and checked yarn quality",
            "similarity_score": 0.87
        }
    ]

    print("Generating AI-Assisted Diagnostic Recommendation using Ollama...")
    print(f"Make sure Ollama is running with model: {OLLAMA_MODEL}")
    diagnosis = synthesize_diagnosis(
        entities=sample_entities,
        anomaly_findings=sample_anomalies,
        historical_incidents=sample_incidents
    )

    print(format_diagnosis(diagnosis))
    print("\n" + "=" * 80)
    print("RAW JSON OUTPUT")
    print("=" * 80)
    print(json.dumps(diagnosis, indent=2))