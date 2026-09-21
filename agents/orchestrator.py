"""
Orchestrator: Chains all 4 agents in sequence
Agent 1 (extract entities) -> Agent 2 (find anomalies) -> Agent 3 (retrieve cases) -> Agent 4 (synthesize)
Passes a growing JSON payload between agents and returns the final diagnosis.
"""

import json
import sys
import os

# Add parent directory to path to import agents
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from agents.agent1_intake import extract_entities
from agents.agent2_anomaly import detect_anomalies, compute_correlations
from agents.agent3_retrieval import retrieve
from agents.agent4_synthesis import synthesize_diagnosis, format_diagnosis


def run_pipeline(query: str, production_df=None, metric_name: str = None, top_k: int = 3) -> dict:
    """
    Run the complete pipeline chaining all 4 agents.
    
    Args:
        query: Natural language query about an incident (e.g., 
               "Why is Loom 12 producing inconsistent elongation in Batch B102?")
        production_df: DataFrame with production readings (optional, for Agent 2)
        metric_name: Name of metric to analyze for anomalies (optional, for Agent 2)
        top_k: Number of similar incidents to retrieve (default: 3, for Agent 3)
    
    Returns:
        Complete pipeline result with all agent outputs and final diagnosis
    """
    # Initialize the payload
    payload = {
        "query": query,
        "agent_outputs": {}
    }
    
    print("=" * 80)
    print("STARTING INCIDENT DIAGNOSIS PIPELINE")
    print("=" * 80)
    print(f"Query: {query}\n")
    
    # ── Agent 1: Extract Entities ────────────────────────────────────────────────
    print("Step 1: Agent 1 - Extracting entities from query...")
    try:
        entities = extract_entities(query)
        payload["agent_outputs"]["agent1_intake"] = entities
        print(f"✓ Entities extracted: {json.dumps(entities, indent=2)}\n")
    except Exception as e:
        payload["agent_outputs"]["agent1_intake"] = {"error": str(e)}
        print(f"✗ Agent 1 failed: {e}\n")
        # Don't return early - continue with other agents
    
    # ── Agent 2: Detect Anomalies ─────────────────────────────────────────────────
    print("Step 2: Agent 2 - Detecting anomalies in production data...")
    if production_df is not None and metric_name is not None:
        try:
            anomaly_results = detect_anomalies(production_df, metric_name)
            payload["agent_outputs"]["agent2_anomaly"] = anomaly_results
            print(f"✓ Anomalies detected: {anomaly_results['anomaly_count']} found\n")
        except Exception as e:
            payload["agent_outputs"]["agent2_anomaly"] = {"error": str(e)}
            print(f"✗ Agent 2 failed: {e}\n")
    else:
        payload["agent_outputs"]["agent2_anomaly"] = {
            "note": "No production data provided, skipping anomaly detection"
        }
        print("⊘ Skipping Agent 2 (no production data provided)\n")
    
    # ── Agent 3: Retrieve Historical Incidents ────────────────────────────────────
    print("Step 3: Agent 3 - Retrieving similar historical incidents...")
    try:
        # Use the extracted entities to build a search query
        search_query_parts = []
        if entities.get("defect_symptom"):
            search_query_parts.append(entities["defect_symptom"])
        if entities.get("metric"):
            search_query_parts.append(entities["metric"])
        if entities.get("loom_id"):
            search_query_parts.append(f"loom {entities['loom_id']}")
        
        search_query = " ".join(search_query_parts) if search_query_parts else query
        
        historical_incidents = retrieve(search_query, top_k=top_k)
        payload["agent_outputs"]["agent3_retrieval"] = historical_incidents
        print(f"✓ Retrieved {len(historical_incidents)} similar incidents\n")
    except Exception as e:
        payload["agent_outputs"]["agent3_retrieval"] = {"error": str(e)}
        print(f"✗ Agent 3 failed: {e}\n")
        return payload
    
    # ── Agent 4: Synthesize Diagnosis ────────────────────────────────────────────
    print("Step 4: Agent 4 - Synthesizing AI-assisted diagnosis...")
    try:
        anomaly_findings = payload["agent_outputs"].get("agent2_anomaly")
        if "error" in anomaly_findings or "note" in anomaly_findings:
            anomaly_findings = None
        
        diagnosis = synthesize_diagnosis(
            entities=entities,
            anomaly_findings=anomaly_findings,
            historical_incidents=historical_incidents
        )
        payload["agent_outputs"]["agent4_synthesis"] = diagnosis
        print("✓ Diagnosis synthesized\n")
    except Exception as e:
        payload["agent_outputs"]["agent4_synthesis"] = {"error": str(e)}
        print(f"✗ Agent 4 failed: {e}\n")
        return payload
    
    # ── Final Output ─────────────────────────────────────────────────────────────
    print("=" * 80)
    print("PIPELINE COMPLETE")
    print("=" * 80)
    
    return payload


def format_pipeline_result(payload: dict) -> str:
    """
    Format the complete pipeline result for human-readable output.
    
    Args:
        payload: Complete pipeline result from run_pipeline()
    
    Returns:
        Formatted string representation
    """
    output = []
    output.append("=" * 80)
    output.append("COMPLETE INCIDENT DIAGNOSIS PIPELINE RESULT")
    output.append("=" * 80)
    output.append(f"\nOriginal Query: {payload['query']}\n")
    
    # Agent 1 Output
    output.append("─" * 80)
    output.append("AGENT 1: ENTITY EXTRACTION")
    output.append("─" * 80)
    agent1 = payload["agent_outputs"]["agent1_intake"]
    if "error" in agent1:
        output.append(f"Error: {agent1['error']}")
    else:
        output.append(json.dumps(agent1, indent=2))
    
    # Agent 2 Output
    output.append("\n" + "─" * 80)
    output.append("AGENT 2: ANOMALY DETECTION")
    output.append("─" * 80)
    agent2 = payload["agent_outputs"].get("agent2_anomaly")
    if agent2 is None:
        output.append("Skipped (no production data provided)")
    elif "error" in agent2:
        output.append(f"Error: {agent2['error']}")
    elif "note" in agent2:
        output.append(agent2["note"])
    else:
        output.append(f"Metric: {agent2.get('metric', 'N/A')}")
        output.append(f"Total Readings: {agent2.get('total_readings', 'N/A')}")
        output.append(f"Anomalies Found: {agent2.get('anomaly_count', 'N/A')}")
        output.append(f"Anomaly Percentage: {agent2.get('anomaly_percentage', 'N/A')}%")
    
    # Agent 3 Output
    output.append("\n" + "─" * 80)
    output.append("AGENT 3: HISTORICAL INCIDENT RETRIEVAL")
    output.append("─" * 80)
    agent3 = payload["agent_outputs"]["agent3_retrieval"]
    if "error" in agent3:
        output.append(f"Error: {agent3['error']}")
    else:
        output.append(f"Retrieved {len(agent3)} similar incidents:")
        for i, incident in enumerate(agent3, 1):
            output.append(f"\n  {i}. Incident ID: {incident['incident_id']}")
            output.append(f"     Reason: {incident['reason']}")
            output.append(f"     Similarity: {incident['similarity_score']:.3f}")
    
    # Agent 4 Output
    output.append("\n" + "─" * 80)
    output.append("AGENT 4: AI-ASSISTED DIAGNOSIS")
    output.append("─" * 80)
    agent4 = payload["agent_outputs"]["agent4_synthesis"]
    if "error" in agent4:
        output.append(f"Error: {agent4['error']}")
    else:
        output.append(format_diagnosis(agent4))
    
    return "\n".join(output)


# ── Main Block: Test Query ───────────────────────────────────────────────────────
if __name__ == "__main__":
    # Test query
    test_query = "Why is Loom 12 producing inconsistent elongation in Batch B102?"
    
    # Run the pipeline without production data (Agent 2 will be skipped)
    # In production, you would pass a DataFrame with production readings
    result = run_pipeline(
        query=test_query,
        production_df=None,  # Pass a DataFrame here to enable Agent 2
        metric_name=None,    # Specify metric name to enable Agent 2
        top_k=3
    )
    
    # Display formatted result
    print(format_pipeline_result(result))
    
    # Also save raw JSON to file
    with open("pipeline_output.json", "w") as f:
        json.dump(result, f, indent=2)
    print("\n" + "=" * 80)
    print("Raw output saved to: pipeline_output.json")
