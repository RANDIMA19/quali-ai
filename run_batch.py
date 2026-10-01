import subprocess, sys

# usage: python run_batch.py <ClassName> <batch_size> <batch_number>
# example: python run_batch.py TestAgent1Intake 6 1
cls, size, num = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])

out = subprocess.run(
    [sys.executable, "-m", "pytest", "test_quali_ai.py", "--collect-only", "-q", "-k", cls],
    capture_output=True, text=True,
).stdout
ids = [l for l in out.splitlines() if l.startswith("test_quali_ai.py::")]

total = -(-len(ids) // size)
batch = ids[(num - 1) * size : num * size]
print(f"{cls}: batch {num} of {total} ({len(ids)} tests in total)\n")
subprocess.run([sys.executable, "-m", "pytest", "-v", "-rxX", *batch])
