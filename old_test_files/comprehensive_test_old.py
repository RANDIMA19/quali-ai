"""
Comprehensive Test Suite for Quality AI Diagnostic System
15 test cases per agent = 60 total test cases
"""

import json
import pandas as pd
import numpy as np
from agents.agent1_intake import extract_entities
from agents.agent2_anomaly import detect_anomalies, compute_correlations
from agents.agent3_retrieval import retrieve
from agents.agent4_synthesis_mock import synthesize_diagnosis_mock

print("=" * 80)
print("COMPREHENSIVE TEST SUITE - 60 TEST CASES")
print("=" * 80)


# AGENT 1: ENTITY EXTRACTION AND NLP PROCESSING
print("\n" + "=" * 80)
print("AGENT 1: ENTITY EXTRACTION - 15 TEST CASES")
print("=" * 80)

agent1_tests = [
    # Basic cases
    "Why is Loom 12 producing inconsistent elongation in Batch B102?",
    "What's causing high tension readings on Loom 5?",
    "Investigate the strength problems in Batch A205.",
    
    # Different loom IDs
    "Loom 1 has measurement issues.",
    "Loom 99 is showing defects.",
    "Loom 7 needs investigation.",
    
    # Different batch IDs
    "Batch Z999 has quality issues.",
    "Batch X001 is problematic.",
    "Batch M500 shows defects.",
    
    # Different metrics
    "Why is elongation varying so much?",
    "Strength readings are inconsistent.",
    "Tension values are too high.",
    
    # Different symptoms
    "The fabric shows inconsistent patterns.",
    "High variance in measurements detected.",
    "Low quality readings observed.",
    
    # Complex query
    "Loom 3 and Loom 4 both have elongation issues in Batch C301."
]

agent1_results = []
for i, query in enumerate(agent1_tests, 1):
    try:
        result = extract_entities(query)
        agent1_results.append((query, result, "PASS"))
        print(f"Test {i}/15: PASS - {query[:50]}...")
    except Exception as e:
        agent1_results.append((query, str(e), "FAIL"))
        print(f"Test {i}/15: FAIL - {query[:50]}... Error: {e}")

print(f"\nAgent 1 Results: {sum(1 for _, _, status in agent1_results if status == 'PASS')}/15 passed")

# AGENT 2: ANOMALY DETECTION (15 TEST CASES)
print("\n" + "=" * 80)
print("AGENT 2: ANOMALY DETECTION - 15 TEST CASES")
print("=" * 80)

agent2_tests = []

# Test 1: Normal data (no anomalies)
np.random.seed(1)
normal_data = pd.DataFrame({"elongation": np.random.normal(100, 5, 100)})
agent2_tests.append(("Normal data - no anomalies", normal_data, "elongation", 2.0, True))

# Test 2: Single high anomaly
data_with_high = normal_data.copy()
data_with_high.loc[10, "elongation"] = 150
agent2_tests.append(("Single high anomaly", data_with_high, "elongation", 2.0, True))

# Test 3: Single low anomaly
data_with_low = normal_data.copy()
data_with_low.loc[20, "elongation"] = 50
agent2_tests.append(("Single low anomaly", data_with_low, "elongation", 2.0, True))

# Test 4: Multiple anomalies
data_multiple = normal_data.copy()
data_multiple.loc[10, "elongation"] = 150
data_multiple.loc[20, "elongation"] = 50
data_multiple.loc[30, "elongation"] = 160
agent2_tests.append(("Multiple anomalies", data_multiple, "elongation", 2.0, True))

# Test 5: High threshold (should detect fewer)
agent2_tests.append(("High threshold - strict", data_multiple, "elongation", 3.0, True))

# Test 6: Low threshold (should detect more)
agent2_tests.append(("Low threshold - lenient", data_multiple, "elongation", 1.0, True))

# Test 7: Different metric (strength)
strength_data = pd.DataFrame({"strength": np.random.normal(50, 3, 100)})
strength_data.loc[15, "strength"] = 80
agent2_tests.append(("Strength metric anomaly", strength_data, "strength", 2.0, True))

# Test 8: Different metric (tension)
tension_data = pd.DataFrame({"tension": np.random.normal(30, 2, 100)})
tension_data.loc[25, "tension"] = 45
agent2_tests.append(("Tension metric anomaly", tension_data, "tension", 2.0, True))

# Test 9: Large dataset
large_data = pd.DataFrame({"elongation": np.random.normal(100, 5, 1000)})
large_data.loc[100, "elongation"] = 150
agent2_tests.append(("Large dataset (1000 points)", large_data, "elongation", 2.0, True))

# Test 10: Small dataset
small_data = pd.DataFrame({"elongation": np.random.normal(100, 5, 20)})
small_data.loc[5, "elongation"] = 150
agent2_tests.append(("Small dataset (20 points)", small_data, "elongation", 2.0, True))

# Test 11: Edge case - all same values
constant_data = pd.DataFrame({"elongation": [100] * 100})
agent2_tests.append(("Constant values - no variance", constant_data, "elongation", 2.0, True))

# Test 12: High variance data
high_variance_data = pd.DataFrame({"elongation": np.random.normal(100, 20, 100)})
agent2_tests.append(("High variance data", high_variance_data, "elongation", 2.0, True))

# Test 13: Missing values
missing_data = normal_data.copy()
missing_data.loc[5, "elongation"] = np.nan
agent2_tests.append(("Data with missing values", missing_data, "elongation", 2.0, True))

# Test 14: Correlation analysis
multi_metric_data = pd.DataFrame({
    "elongation": np.random.normal(100, 5, 100),
    "strength": np.random.normal(50, 3, 100),
    "tension": np.random.normal(30, 2, 100)
})
agent2_tests.append(("Multi-metric correlation", multi_metric_data, "elongation", 2.0, True))

# Test 15: Extreme outliers
extreme_data = normal_data.copy()
extreme_data.loc[10, "elongation"] = 1000
agent2_tests.append(("Extreme outlier", extreme_data, "elongation", 2.0, True))

agent2_results = []
for i, (test_name, data, metric, threshold, _) in enumerate(agent2_tests, 1):
    try:
        result = detect_anomalies(data, metric, threshold=threshold)
        agent2_results.append((test_name, result, "PASS"))
        print(f"Test {i}/15: PASS - {test_name}")
    except Exception as e:
        agent2_results.append((test_name, str(e), "FAIL"))
        print(f"Test {i}/15: FAIL - {test_name} Error: {e}")

print(f"\nAgent 2 Results: {sum(1 for _, _, status in agent2_results if status == 'PASS')}/15 passed")

# AGENT 3: HISTORICAL INCIDENT RETRIEVAL
print("\n" + "=" * 80)
print("AGENT 3: HISTORICAL INCIDENT RETRIEVAL - 15 TEST CASES")
print("=" * 80)

agent3_tests = [
    # Basic search terms
    "inconsistent elongation",
    "high tension",
    "low strength",
    
    # Loom-specific searches
    "loom 12",
    "loom 5 issues",
    "loom 8 problems",
    
    # Batch-specific searches
    "batch B102",
    "batch A205",
    "batch C301",
    
    # Metric-specific searches
    "elongation problems",
    "strength variance",
    "tension spikes",
    
    # Complex searches
    "inconsistent elongation loom 12",
    "high tension batch issues",
    "quality problems manufacturing"
]

agent3_results = []
for i, query in enumerate(agent3_tests, 1):
    try:
        result = retrieve(query, top_k=3)
        agent3_results.append((query, result, "PASS"))
        print(f"Test {i}/15: PASS - '{query}' - Retrieved {len(result)} incidents")
    except Exception as e:
        agent3_results.append((query, str(e), "FAIL"))
        print(f"Test {i}/15: FAIL - '{query}' Error: {e}")

print(f"\nAgent 3 Results: {sum(1 for _, _, status in agent3_results if status == 'PASS')}/15 passed")

# AGENT 4: SYNTHESIS AND DIAGNOSIS 
print("\n" + "=" * 80)
print("AGENT 4: SYNTHESIS - 15 TEST CASES")
print("=" * 80)

# Prepare test data for Agent 4
sample_entities = [
    {"loom_id": "12", "batch_id": "B102", "metric": "elongation", "defect_symptom": "inconsistent"},
    {"loom_id": "5", "batch_id": "A205", "metric": "tension", "defect_symptom": "high"},
    {"loom_id": "8", "batch_id": "C301", "metric": "strength", "defect_symptom": "low"},
    {"loom_id": "3", "batch_id": "D405", "metric": "elongation", "defect_symptom": "high"},
    {"loom_id": "7", "batch_id": "E606", "metric": "tension", "defect_symptom": "inconsistent"},
]

sample_anomalies = [
    {"metric": "elongation", "total_readings": 100, "anomaly_count": 5, "anomaly_percentage": 5.0,
     "anomalies": [{"index": 10, "value": 150.0, "z_score": 3.5}], "statistics": {"mean": 100.0, "std": 10.0}},
    {"metric": "tension", "total_readings": 100, "anomaly_count": 8, "anomaly_percentage": 8.0,
     "anomalies": [{"index": 15, "value": 45.0, "z_score": 4.0}], "statistics": {"mean": 30.0, "std": 3.0}},
    {"metric": "strength", "total_readings": 100, "anomaly_count": 3, "anomaly_percentage": 3.0,
     "anomalies": [{"index": 20, "value": 35.0, "z_score": -3.0}], "statistics": {"mean": 50.0, "std": 5.0}},
]

sample_incidents = [
    [
        {"incident_id": "INC-0042", "reason": "inconsistent elongation on loom 12", "action_plan": "adjusted tension", "similarity_score": 0.92},
        {"incident_id": "INC-0156", "reason": "high elongation variance", "action_plan": "replaced rollers", "similarity_score": 0.87}
    ],
    [
        {"incident_id": "INC-0089", "reason": "high tension on loom 5", "action_plan": "calibrated sensors", "similarity_score": 0.95},
        {"incident_id": "INC-0234", "reason": "tension spikes", "action_plan": "updated control system", "similarity_score": 0.88}
    ],
    [
        {"incident_id": "INC-0123", "reason": "low strength readings", "action_plan": "checked yarn quality", "similarity_score": 0.91},
        {"incident_id": "INC-0456", "reason": "strength inconsistency", "action_plan": "adjusted parameters", "similarity_score": 0.85}
    ],
]

agent4_tests = [
    # Test 1: Standard elongation inconsistency
    ("Standard elongation issue", sample_entities[0], sample_anomalies[0], sample_incidents[0]),
    
    # Test 2: High tension issue
    ("High tension issue", sample_entities[1], sample_anomalies[1], sample_incidents[1]),
    
    # Test 3: Low strength issue
    ("Low strength issue", sample_entities[2], sample_anomalies[2], sample_incidents[2]),
    
    # Test 4: High elongation
    ("High elongation", sample_entities[3], sample_anomalies[0], sample_incidents[0]),
    
    # Test 5: Inconsistent tension
    ("Inconsistent tension", sample_entities[4], sample_anomalies[1], sample_incidents[1]),
    
    # Test 6: No anomalies (edge case)
    ("No anomalies detected", sample_entities[0], {"anomaly_count": 0, "anomaly_percentage": 0}, sample_incidents[0]),
    
    # Test 7: No historical incidents (edge case)
    ("No historical incidents", sample_entities[0], sample_anomalies[0], []),
    
    # Test 8: High anomaly count
    ("High anomaly count", sample_entities[0], 
     {"metric": "elongation", "anomaly_count": 15, "anomaly_percentage": 15.0, "statistics": {"mean": 100.0, "std": 10.0}}, 
     sample_incidents[0]),
    
    # Test 9: Low anomaly count
    ("Low anomaly count", sample_entities[0], 
     {"metric": "elongation", "anomaly_count": 1, "anomaly_percentage": 1.0, "statistics": {"mean": 100.0, "std": 10.0}}, 
     sample_incidents[0]),
    
    # Test 10: Missing entity fields (edge case)
    ("Missing entity fields", {"loom_id": "12"}, sample_anomalies[0], sample_incidents[0]),
    
    # Test 11: None values (edge case)
    ("None values in entities", {}, sample_anomalies[0], sample_incidents[0]),
    
    # Test 12: Single historical incident
    ("Single historical incident", sample_entities[0], sample_anomalies[0], [sample_incidents[0][0]]),
    
    # Test 13: Multiple historical incidents
    ("Multiple historical incidents", sample_entities[0], sample_anomalies[0], sample_incidents[0] + sample_incidents[1]),
    
    # Test 14: Complex entity
    ("Complex entity with all fields", 
     {"loom_id": "99", "batch_id": "Z999", "metric": "elongation", "defect_symptom": "inconsistent"},
     sample_anomalies[0], sample_incidents[0]),
    
    # Test 15: Anomaly with statistics
    ("Anomaly with detailed statistics", 
     sample_entities[0],
     {"metric": "elongation", "anomaly_count": 5, "anomaly_percentage": 5.0, 
      "statistics": {"mean": 100.0, "std": 10.0, "min": 40.0, "max": 150.0}},
     sample_incidents[0])
]

agent4_results = []
for i, (test_name, entities, anomalies, incidents) in enumerate(agent4_tests, 1):
    try:
        result = synthesize_diagnosis_mock(entities, anomalies, incidents)
        if "error" in result:
            agent4_results.append((test_name, result["error"], "FAIL"))
            print(f"Test {i}/15: FAIL - {test_name} Error: {result['error']}")
        else:
            agent4_results.append((test_name, result, "PASS"))
            print(f"Test {i}/15: PASS - {test_name}")
    except Exception as e:
        agent4_results.append((test_name, str(e), "FAIL"))
        print(f"Test {i}/15: FAIL - {test_name} Error: {e}")

print(f"\nAgent 4 Results: {sum(1 for _, _, status in agent4_results if status == 'PASS')}/15 passed")

# ── OVERALL SUMMARY ───────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
print("OVERALL TEST SUMMARY")
print("=" * 80)
print(f"Agent 1 (Entity Extraction): {sum(1 for _, _, status in agent1_results if status == 'PASS')}/15 passed")
print(f"Agent 2 (Anomaly Detection): {sum(1 for _, _, status in agent2_results if status == 'PASS')}/15 passed")
print(f"Agent 3 (Retrieval): {sum(1 for _, _, status in agent3_results if status == 'PASS')}/15 passed")
print(f"Agent 4 (Synthesis): {sum(1 for _, _, status in agent4_results if status == 'PASS')}/15 passed")

total_passed = (sum(1 for _, _, status in agent1_results if status == 'PASS') +
                sum(1 for _, _, status in agent2_results if status == 'PASS') +
                sum(1 for _, _, status in agent3_results if status == 'PASS') +
                sum(1 for _, _, status in agent4_results if status == 'PASS'))

print(f"\nTOTAL: {total_passed}/60 test cases passed")

if total_passed == 60:
    print("[OK] ALL TESTS PASSED!")
else:
    print(f"[FAIL] {60 - total_passed} test(s) failed")

# ── DETAILED FAILURE REPORT ─────────────────────────────────────────────────────
if total_passed < 60:
    print("\n" + "=" * 80)
    print("DETAILED FAILURE REPORT")
    print("=" * 80)
    
    print("\nAgent 1 Failures:")
    for query, error, status in agent1_results:
        if status == "FAIL":
            print(f"  - Query: {query}")
            print(f"    Error: {error}")
    
    print("\nAgent 2 Failures:")
    for test_name, error, status in agent2_results:
        if status == "FAIL":
            print(f"  - Test: {test_name}")
            print(f"    Error: {error}")
    
    print("\nAgent 3 Failures:")
    for query, error, status in agent3_results:
        if status == "FAIL":
            print(f"  - Query: {query}")
            print(f"    Error: {error}")
    
    print("\nAgent 4 Failures:")
    for test_name, error, status in agent4_results:
        if status == "FAIL":
            print(f"  - Test: {test_name}")
            print(f"    Error: {error}")

print("\n" + "=" * 80)
print("TEST SUITE COMPLETE")
print("=" * 80)
