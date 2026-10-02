"""
Batch Test Runner for Quality AI Diagnostic System
Allows running tests one by one or in batches for screenshot purposes
"""

import json
import pandas as pd
import numpy as np
import time
from agents.agent1_intake import extract_entities
from agents.agent2_anomaly import detect_anomalies, compute_correlations
from agents.agent3_retrieval import retrieve
from agents.agent4_synthesis_mock import synthesize_diagnosis_mock

print("=" * 80)
print("BATCH TEST RUNNER - FOR SCREENSHOT PURPOSES")
print("=" * 80)

# ── TEST CASE DEFINITIONS ───────────────────────────────────────────────────────

# Agent 1: Entity Extraction Test Cases
agent1_tests = [
    "Why is Loom 12 producing inconsistent elongation in Batch B102?",
    "What's causing high tension readings on Loom 5?",
    "Investigate the strength problems in Batch A205.",
    "Loom 1 has measurement issues.",
    "Loom 99 is showing defects.",
    "Loom 7 needs investigation.",
    "Batch Z999 has quality issues.",
    "Batch X001 is problematic.",
    "Batch M500 shows defects.",
    "Why is elongation varying so much?",
    "Strength readings are inconsistent.",
    "Tension values are too high.",
    "The fabric shows inconsistent patterns.",
    "High variance in measurements detected.",
    "Low quality readings observed."
]

# Agent 2: Anomaly Detection Test Cases
def create_agent2_tests():
    tests = []
    
    # Test 1: Normal data
    np.random.seed(1)
    normal_data = pd.DataFrame({"elongation": np.random.normal(100, 5, 100)})
    tests.append(("Normal data - no anomalies", normal_data, "elongation", 2.0))
    
    # Test 2: Single high anomaly
    data_with_high = normal_data.copy()
    data_with_high.loc[10, "elongation"] = 150
    tests.append(("Single high anomaly", data_with_high, "elongation", 2.0))
    
    # Test 3: Single low anomaly
    data_with_low = normal_data.copy()
    data_with_low.loc[20, "elongation"] = 50
    tests.append(("Single low anomaly", data_with_low, "elongation", 2.0))
    
    # Test 4: Multiple anomalies
    data_multiple = normal_data.copy()
    data_multiple.loc[10, "elongation"] = 150
    data_multiple.loc[20, "elongation"] = 50
    data_multiple.loc[30, "elongation"] = 160
    tests.append(("Multiple anomalies", data_multiple, "elongation", 2.0))
    
    # Test 5: High threshold
    tests.append(("High threshold - strict", data_multiple, "elongation", 3.0))
    
    # Test 6: Low threshold
    tests.append(("Low threshold - lenient", data_multiple, "elongation", 1.0))
    
    # Test 7: Strength metric
    strength_data = pd.DataFrame({"strength": np.random.normal(50, 3, 100)})
    strength_data.loc[15, "strength"] = 80
    tests.append(("Strength metric anomaly", strength_data, "strength", 2.0))
    
    # Test 8: Tension metric
    tension_data = pd.DataFrame({"tension": np.random.normal(30, 2, 100)})
    tension_data.loc[25, "tension"] = 45
    tests.append(("Tension metric anomaly", tension_data, "tension", 2.0))
    
    # Test 9: Large dataset
    large_data = pd.DataFrame({"elongation": np.random.normal(100, 5, 1000)})
    large_data.loc[100, "elongation"] = 150
    tests.append(("Large dataset (1000 points)", large_data, "elongation", 2.0))
    
    # Test 10: Small dataset
    small_data = pd.DataFrame({"elongation": np.random.normal(100, 5, 20)})
    small_data.loc[5, "elongation"] = 150
    tests.append(("Small dataset (20 points)", small_data, "elongation", 2.0))
    
    # Test 11: Constant values
    constant_data = pd.DataFrame({"elongation": [100] * 100})
    tests.append(("Constant values - no variance", constant_data, "elongation", 2.0))
    
    # Test 12: High variance
    high_variance_data = pd.DataFrame({"elongation": np.random.normal(100, 20, 100)})
    tests.append(("High variance data", high_variance_data, "elongation", 2.0))
    
    # Test 13: Missing values
    missing_data = normal_data.copy()
    missing_data.loc[5, "elongation"] = np.nan
    tests.append(("Data with missing values", missing_data, "elongation", 2.0))
    
    # Test 14: Multi-metric correlation
    multi_metric_data = pd.DataFrame({
        "elongation": np.random.normal(100, 5, 100),
        "strength": np.random.normal(50, 3, 100),
        "tension": np.random.normal(30, 2, 100)
    })
    tests.append(("Multi-metric correlation", multi_metric_data, "elongation", 2.0))
    
    # Test 15: Extreme outlier
    extreme_data = normal_data.copy()
    extreme_data.loc[10, "elongation"] = 1000
    tests.append(("Extreme outlier", extreme_data, "elongation", 2.0))
    
    return tests

# Agent 3: Retrieval Test Cases
agent3_tests = [
    "inconsistent elongation",
    "high tension",
    "low strength",
    "loom 12",
    "loom 5 issues",
    "loom 8 problems",
    "batch B102",
    "batch A205",
    "batch C301",
    "elongation problems",
    "strength variance",
    "tension spikes",
    "inconsistent elongation loom 12",
    "high tension batch issues",
    "quality problems manufacturing"
]

# Agent 4: Synthesis Test Cases
def create_agent4_tests():
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

    tests = [
        ("Standard elongation issue", sample_entities[0], sample_anomalies[0], sample_incidents[0]),
        ("High tension issue", sample_entities[1], sample_anomalies[1], sample_incidents[1]),
        ("Low strength issue", sample_entities[2], sample_anomalies[2], sample_incidents[2]),
        ("High elongation", sample_entities[3], sample_anomalies[0], sample_incidents[0]),
        ("Inconsistent tension", sample_entities[4], sample_anomalies[1], sample_incidents[1]),
        ("No anomalies detected", sample_entities[0], {"anomaly_count": 0, "anomaly_percentage": 0}, sample_incidents[0]),
        ("No historical incidents", sample_entities[0], sample_anomalies[0], []),
        ("High anomaly count", sample_entities[0], 
         {"metric": "elongation", "anomaly_count": 15, "anomaly_percentage": 15.0, "statistics": {"mean": 100.0, "std": 10.0}}, 
         sample_incidents[0]),
        ("Low anomaly count", sample_entities[0], 
         {"metric": "elongation", "anomaly_count": 1, "anomaly_percentage": 1.0, "statistics": {"mean": 100.0, "std": 10.0}}, 
         sample_incidents[0]),
        ("Missing entity fields", {"loom_id": "12"}, sample_anomalies[0], sample_incidents[0]),
        ("None values in entities", {}, sample_anomalies[0], sample_incidents[0]),
        ("Single historical incident", sample_entities[0], sample_anomalies[0], [sample_incidents[0][0]]),
        ("Multiple historical incidents", sample_entities[0], sample_anomalies[0], sample_incidents[0] + sample_incidents[1]),
        ("Complex entity with all fields", 
         {"loom_id": "99", "batch_id": "Z999", "metric": "elongation", "defect_symptom": "inconsistent"},
         sample_anomalies[0], sample_incidents[0]),
        ("Anomaly with detailed statistics", 
         sample_entities[0],
         {"metric": "elongation", "anomaly_count": 5, "anomaly_percentage": 5.0, 
          "statistics": {"mean": 100.0, "std": 10.0, "min": 40.0, "max": 150.0}},
         sample_incidents[0])
    ]
    return tests

# ── TEST EXECUTION FUNCTIONS ─────────────────────────────────────────────────────

def run_agent1_test(test_case, test_num):
    """Run a single Agent 1 test"""
    print(f"\n{'='*80}")
    print(f"AGENT 1 - TEST {test_num}/15: Entity Extraction")
    print(f"{'='*80}")
    print(f"Query: {test_case}")
    print(f"\nRunning test...")
    
    try:
        result = extract_entities(test_case)
        print(f"\n[SUCCESS] Result:")
        print(json.dumps(result, indent=2))
        return True, result
    except Exception as e:
        print(f"\n[FAILED] Error: {e}")
        return False, str(e)

def run_agent2_test(test_case, test_num):
    """Run a single Agent 2 test"""
    test_name, data, metric, threshold = test_case
    print(f"\n{'='*80}")
    print(f"AGENT 2 - TEST {test_num}/15: Anomaly Detection")
    print(f"{'='*80}")
    print(f"Test: {test_name}")
    print(f"Metric: {metric}, Threshold: {threshold}")
    print(f"Data shape: {data.shape}")
    print(f"\nRunning test...")
    
    try:
        result = detect_anomalies(data, metric, threshold=threshold)
        print(f"\n[SUCCESS] Result:")
        print(f"Anomalies found: {result['anomaly_count']}")
        print(f"Anomaly percentage: {result['anomaly_percentage']}%")
        return True, result
    except Exception as e:
        print(f"\n[FAILED] Error: {e}")
        return False, str(e)

def run_agent3_test(test_case, test_num):
    """Run a single Agent 3 test"""
    print(f"\n{'='*80}")
    print(f"AGENT 3 - TEST {test_num}/15: Historical Incident Retrieval")
    print(f"{'='*80}")
    print(f"Query: {test_case}")
    print(f"\nRunning test...")
    
    try:
        result = retrieve(test_case, top_k=3)
        print(f"\n[SUCCESS] Retrieved {len(result)} incidents:")
        for i, incident in enumerate(result, 1):
            print(f"{i}. {incident['incident_id']}: {incident['reason']} (similarity: {incident['similarity_score']:.3f})")
        return True, result
    except Exception as e:
        print(f"\n[FAILED] Error: {e}")
        return False, str(e)

def run_agent4_test(test_case, test_num):
    """Run a single Agent 4 test"""
    test_name, entities, anomalies, incidents = test_case
    print(f"\n{'='*80}")
    print(f"AGENT 4 - TEST {test_num}/15: Synthesis")
    print(f"{'='*80}")
    print(f"Test: {test_name}")
    print(f"Entities: {json.dumps(entities, indent=2)}")
    print(f"\nRunning test...")
    
    try:
        result = synthesize_diagnosis_mock(entities, anomalies, incidents)
        if "error" in result:
            print(f"\n[FAILED] Error: {result['error']}")
            return False, result['error']
        else:
            print(f"\n[SUCCESS] Diagnosis generated:")
            print(f"Label: {result['label']}")
            if 'diagnosis' in result:
                diag = result['diagnosis']
                if 'root_cause_analysis' in diag:
                    rca = diag['root_cause_analysis']
                    print(f"Primary cause: {rca.get('primary_cause', 'N/A')}")
                    print(f"Severity: {rca.get('severity', 'N/A')}")
            return True, result
    except Exception as e:
        print(f"\n[FAILED] Error: {e}")
        return False, str(e)

# ── BATCH RUNNING FUNCTIONS ─────────────────────────────────────────────────────

def run_tests_batch(agent_tests, agent_name, test_function, batch_size=2):
    """Run tests in batches with pauses for screenshots"""
    total_tests = len(agent_tests)
    print(f"\n{'='*80}")
    print(f"RUNNING {agent_name} TESTS - BATCH SIZE: {batch_size}")
    print(f"Total tests: {total_tests}")
    print(f"{'='*80}")
    
    results = []
    for i in range(0, total_tests, batch_size):
        batch = agent_tests[i:i+batch_size]
        batch_num = (i // batch_size) + 1
        total_batches = (total_tests + batch_size - 1) // batch_size
        
        print(f"\n{'─'*80}")
        print(f"BATCH {batch_num}/{total_batches} (Tests {i+1}-{min(i+batch_size, total_tests)})")
        print(f"{'─'*80}")
        
        for j, test_case in enumerate(batch, i+1):
            success, result = test_function(test_case, j)
            results.append((test_case, success, result))
            
            # Pause between tests in the same batch
            if j < min(i+batch_size, total_tests):
                print(f"\n[PAUSE] Take screenshot if needed...")
                input("Press Enter to continue to next test...")
        
        # Pause between batches
        if i + batch_size < total_tests:
            print(f"\n{'='*80}")
            print(f"[END OF BATCH {batch_num}]")
            print(f"Take screenshots of this batch before continuing...")
            input("Press Enter to continue to next batch...")
            print(f"{'='*80}")
    
    return results

def run_single_tests(agent_tests, agent_name, test_function):
    """Run tests one by one with pauses for screenshots"""
    total_tests = len(agent_tests)
    print(f"\n{'='*80}")
    print(f"RUNNING {agent_name} TESTS - ONE BY ONE")
    print(f"Total tests: {total_tests}")
    print(f"{'='*80}")
    
    results = []
    for i, test_case in enumerate(agent_tests, 1):
        success, result = test_function(test_case, i)
        results.append((test_case, success, result))
        
        # Pause between tests
        if i < total_tests:
            print(f"\n[PAUSE] Take screenshot for test {i} before continuing...")
            input("Press Enter to continue to next test...")
    
    return results

# ── MAIN MENU ─────────────────────────────────────────────────────────────────

def main():
    print("\n" + "="*80)
    print("BATCH TEST RUNNER - MAIN MENU")
    print("="*80)
    print("1. Run Agent 1 tests (Entity Extraction)")
    print("2. Run Agent 2 tests (Anomaly Detection)")
    print("3. Run Agent 3 tests (Retrieval)")
    print("4. Run Agent 4 tests (Synthesis)")
    print("5. Run all agents (sequential)")
    print("6. Exit")
    
    choice = input("\nEnter your choice (1-6): ")
    
    if choice == "1":
        batch_size = int(input("Enter batch size (1 for individual, 2+ for batches): "))
        if batch_size == 1:
            results = run_single_tests(agent1_tests, "AGENT 1", run_agent1_test)
        else:
            results = run_tests_batch(agent1_tests, "AGENT 1", run_agent1_test, batch_size)
        
        passed = sum(1 for _, success, _ in results if success)
        print(f"\nAgent 1 Results: {passed}/{len(results)} passed")
        
    elif choice == "2":
        batch_size = int(input("Enter batch size (1 for individual, 2+ for batches): "))
        agent2_tests = create_agent2_tests()
        if batch_size == 1:
            results = run_single_tests(agent2_tests, "AGENT 2", run_agent2_test)
        else:
            results = run_tests_batch(agent2_tests, "AGENT 2", run_agent2_test, batch_size)
        
        passed = sum(1 for _, success, _ in results if success)
        print(f"\nAgent 2 Results: {passed}/{len(results)} passed")
        
    elif choice == "3":
        batch_size = int(input("Enter batch size (1 for individual, 2+ for batches): "))
        if batch_size == 1:
            results = run_single_tests(agent3_tests, "AGENT 3", run_agent3_test)
        else:
            results = run_tests_batch(agent3_tests, "AGENT 3", run_agent3_test, batch_size)
        
        passed = sum(1 for _, success, _ in results if success)
        print(f"\nAgent 3 Results: {passed}/{len(results)} passed")
        
    elif choice == "4":
        batch_size = int(input("Enter batch size (1 for individual, 2+ for batches): "))
        agent4_tests = create_agent4_tests()
        if batch_size == 1:
            results = run_single_tests(agent4_tests, "AGENT 4", run_agent4_test)
        else:
            results = run_tests_batch(agent4_tests, "AGENT 4", run_agent4_test, batch_size)
        
        passed = sum(1 for _, success, _ in results if success)
        print(f"\nAgent 4 Results: {passed}/{len(results)} passed")
        
    elif choice == "5":
        batch_size = int(input("Enter batch size (1 for individual, 2+ for batches): "))
        
        print("\nRunning all agents sequentially...")
        
        # Agent 1
        if batch_size == 1:
            agent1_results = run_single_tests(agent1_tests, "AGENT 1", run_agent1_test)
        else:
            agent1_results = run_tests_batch(agent1_tests, "AGENT 1", run_agent1_test, batch_size)
        
        # Agent 2
        agent2_tests = create_agent2_tests()
        if batch_size == 1:
            agent2_results = run_single_tests(agent2_tests, "AGENT 2", run_agent2_test)
        else:
            agent2_results = run_tests_batch(agent2_tests, "AGENT 2", run_agent2_test, batch_size)
        
        # Agent 3
        if batch_size == 1:
            agent3_results = run_single_tests(agent3_tests, "AGENT 3", run_agent3_test)
        else:
            agent3_results = run_tests_batch(agent3_tests, "AGENT 3", run_agent3_test, batch_size)
        
        # Agent 4
        agent4_tests = create_agent4_tests()
        if batch_size == 1:
            agent4_results = run_single_tests(agent4_tests, "AGENT 4", run_agent4_test)
        else:
            agent4_results = run_tests_batch(agent4_tests, "AGENT 4", run_agent4_test, batch_size)
        
        # Summary
        total_passed = (sum(1 for _, success, _ in agent1_results if success) +
                      sum(1 for _, success, _ in agent2_results if success) +
                      sum(1 for _, success, _ in agent3_results if success) +
                      sum(1 for _, success, _ in agent4_results if success))
        
        print(f"\n{'='*80}")
        print("FINAL SUMMARY")
        print(f"{'='*80}")
        print(f"Agent 1: {sum(1 for _, success, _ in agent1_results if success)}/15 passed")
        print(f"Agent 2: {sum(1 for _, success, _ in agent2_results if success)}/15 passed")
        print(f"Agent 3: {sum(1 for _, success, _ in agent3_results if success)}/15 passed")
        print(f"Agent 4: {sum(1 for _, success, _ in agent4_results if success)}/15 passed")
        print(f"\nTOTAL: {total_passed}/60 passed")
        
    elif choice == "6":
        print("Exiting...")
        return
    else:
        print("Invalid choice. Please try again.")

if __name__ == "__main__":
    while True:
        main()
        cont = input("\nDo you want to run more tests? (y/n): ")
        if cont.lower() != 'y':
            break
