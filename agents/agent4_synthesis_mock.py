"""
Agent 4: Synthesis - Root-Cause Diagnosis (Mock/Rule-based Version)
Alternative implementation that doesn't require Ollama or external LLM services.
Uses rule-based logic to generate diagnostic recommendations.
"""

import json
from typing import Dict, List, Any


def synthesize_diagnosis_mock(
    entities: dict,
    anomaly_findings: dict,
    historical_incidents: list,
    use_llm: bool = False
) -> dict:
    """
    Synthesize a root-cause diagnosis using rule-based logic instead of LLM.
    
    Args:
        entities: Extracted entities from Agent 1 (loom_id, batch_id, metric, defect_symptom)
        anomaly_findings: Anomaly detection results from Agent 2
        historical_incidents: Retrieved similar incidents from Agent 3
        use_llm: If True, will try to use the original LLM-based approach
    
    Returns:
        JSON dict containing the diagnostic recommendation.
    """
    
    # If user specifically wants LLM, try the original approach
    if use_llm:
        try:
            from agents.agent4_synthesis import synthesize_diagnosis
            return synthesize_diagnosis(entities, anomaly_findings, historical_incidents)
        except Exception as e:
            # Fall back to rule-based if LLM fails
            print(f"LLM approach failed, falling back to rule-based: {e}")
    
    # Rule-based diagnosis logic
    input_summary = {
        "entities": entities,
        "anomaly_count": anomaly_findings.get("anomaly_count", 0) if anomaly_findings else 0,
        "historical_incidents_reviewed": len(historical_incidents)
    }
    
    # Extract key information
    loom_id = entities.get("loom_id") or "unknown"
    batch_id = entities.get("batch_id") or "unknown"
    metric = entities.get("metric") or "unknown"
    symptom = entities.get("defect_symptom") or "unknown"
    
    anomaly_count = anomaly_findings.get("anomaly_count", 0) if anomaly_findings else 0
    anomaly_percentage = anomaly_findings.get("anomaly_percentage", 0) if anomaly_findings else 0
    
    # Determine severity based on anomaly count and percentage
    if anomaly_percentage > 10 or anomaly_count > 10:
        severity = "critical"
    elif anomaly_percentage > 5 or anomaly_count > 5:
        severity = "high"
    elif anomaly_percentage > 2 or anomaly_count > 2:
        severity = "medium"
    else:
        severity = "low"
    
    # Determine primary cause based on patterns
    primary_cause = determine_primary_cause(metric, symptom, historical_incidents)
    
    # Generate contributing factors
    contributing_factors = generate_contributing_factors(metric, symptom, anomaly_findings)
    
    # Build cited incidents from historical data
    cited_incidents = []
    for incident in historical_incidents[:3]:  # Use top 3 most relevant
        relevance_text = f"Similar {symptom or 'quality'} {metric or 'measurement'} issue"
        cited_incidents.append({
            "incident_id": incident.get("incident_id", "unknown"),
            "relevance": relevance_text,
            "similarity_score": incident.get("similarity_score", 0.8)
        })
    
    # Generate recommendations based on diagnosis
    recommendations = generate_recommendations(primary_cause, severity, metric)
    
    # Determine confidence level
    confidence = "high" if len(historical_incidents) > 0 and anomaly_count > 0 else "medium"
    
    diagnosis_data = {
        "root_cause_analysis": {
            "primary_cause": primary_cause,
            "contributing_factors": contributing_factors,
            "severity": severity
        },
        "cited_incidents": cited_incidents,
        "recommendations": recommendations,
        "confidence": confidence,
        "additional_notes": f"Analysis based on {anomaly_count} anomalies detected in {metric or 'unknown metric'} for {symptom or 'unknown symptom'} symptoms on loom {loom_id}, batch {batch_id}."
    }
    
    return {
        "label": "Rule-Based Diagnostic Recommendation",
        "diagnosis": diagnosis_data,
        "input_summary": input_summary,
        "method": "rule_based"
    }


def determine_primary_cause(metric: str, symptom: str, historical_incidents: list) -> str:
    """Determine primary cause based on metric, symptom, and historical patterns."""
    
    # Check for patterns in historical incidents
    if historical_incidents:
        similar_reasons = [inc.get("reason", "") for inc in historical_incidents]
        if "tension" in " ".join(similar_reasons).lower():
            return "Tension setting misalignment"
        elif "calibration" in " ".join(similar_reasons).lower():
            return "Sensor calibration drift"
        elif "roller" in " ".join(similar_reasons).lower():
            return "Roller wear or misalignment"
        elif "yarn" in " ".join(similar_reasons).lower():
            return "Yarn quality inconsistency"
    
    # Default rule-based causes based on metric and symptom
    cause_mapping = {
        "elongation": {
            "inconsistent": "Tension setting variability or sensor calibration drift",
            "high": "Excessive tension or roller friction",
            "low": "Insufficient tension or roller slippage"
        },
        "strength": {
            "inconsistent": "Yarn quality variation or tension fluctuations",
            "high": "Over-tensioning or material defects",
            "low": "Under-tensioning or material weakness"
        },
        "tension": {
            "inconsistent": "Tension control system malfunction",
            "high": "Tension control system overcompensation",
            "low": "Tension control system underperformance"
        }
    }
    
    metric_value = metric if metric else "unknown"
    symptom_value = symptom if symptom else "unknown"
    return cause_mapping.get(metric, {}).get(symptom, f"Unknown cause related to {metric_value} {symptom_value}")


def generate_contributing_factors(metric: str, symptom: str, anomaly_findings: dict) -> List[str]:
    """Generate contributing factors based on analysis."""
    
    factors = []
    
    # Add general factors
    if metric:
        factors.append(f"{metric.capitalize()} measurement variability")
    else:
        factors.append("Measurement variability")
    
    if symptom:
        factors.append(f"{symptom.capitalize()} pattern in production data")
    else:
        factors.append("Pattern in production data")
    
    # Add specific factors based on anomaly findings
    if anomaly_findings:
        anomaly_count = anomaly_findings.get("anomaly_count", 0)
        if anomaly_count > 5:
            factors.append("High frequency of anomalies suggests systemic issue")
        elif anomaly_count > 0:
            factors.append("Intermittent anomalies suggest operational variability")
        
        # Check statistics
        stats = anomaly_findings.get("statistics", {})
        if stats:
            std_dev = stats.get("std", 0)
            mean = stats.get("mean", 0)
            if std_dev > mean * 0.1:  # High coefficient of variation
                factors.append("High process variability detected")
    
    # Add metric-specific factors
    metric_factors = {
        "elongation": ["Potential tension control issues", "Possible sensor drift"],
        "strength": ["Material property variations", "Possible over/under tensioning"],
        "tension": ["Control system calibration", "Mechanical component wear"]
    }
    
    factors.extend(metric_factors.get(metric, []))
    
    return factors[:5]  # Limit to top 5 factors


def generate_recommendations(primary_cause: str, severity: str, metric: str) -> List[dict]:
    """Generate actionable recommendations based on diagnosis."""
    
    recommendations = []
    
    # General recommendations based on severity
    if severity in ["critical", "high"]:
        recommendations.append({
            "action": "Immediate production halt for loom inspection",
            "priority": "immediate",
            "expected_outcome": "Prevent further quality issues and equipment damage"
        })
    
    # Specific recommendations based on cause
    if "tension" in primary_cause.lower():
        recommendations.append({
            "action": "Recalibrate tension control system",
            "priority": "short-term",
            "expected_outcome": "Stabilize tension settings and reduce variability"
        })
        recommendations.append({
            "action": "Inspect tension sensors and actuators",
            "priority": "short-term",
            "expected_outcome": "Identify and replace faulty components"
        })
    
    elif "calibration" in primary_cause.lower():
        recommendations.append({
            "action": "Perform sensor calibration using reference standards",
            "priority": "immediate",
            "expected_outcome": "Restore accurate measurement readings"
        })
        recommendations.append({
            "action": "Schedule regular calibration maintenance",
            "priority": "long-term",
            "expected_outcome": "Prevent future calibration drift"
        })
    
    elif "roller" in primary_cause.lower():
        recommendations.append({
            "action": "Inspect and replace worn rollers",
            "priority": "short-term",
            "expected_outcome": "Eliminate mechanical sources of variability"
        })
        recommendations.append({
            "action": "Check roller alignment and tension",
            "priority": "short-term",
            "expected_outcome": "Ensure consistent material handling"
        })
    
    elif "yarn" in primary_cause.lower():
        recommendations.append({
            "action": "Review yarn quality specifications and supplier consistency",
            "priority": "short-term",
            "expected_outcome": "Identify and address material quality issues"
        })
        recommendations.append({
            "action": "Implement incoming material quality checks",
            "priority": "long-term",
            "expected_outcome": "Prevent future material-related defects"
        })
    
    # Default recommendations
    if not recommendations:
        recommendations.append({
            "action": "Conduct comprehensive equipment inspection",
            "priority": "short-term",
            "expected_outcome": "Identify root cause of variability"
        })
        recommendations.append({
            "action": "Review and adjust process parameters",
            "priority": "short-term",
            "expected_outcome": "Optimize production settings"
        })
    
    # Add monitoring recommendation
    recommendations.append({
        "action": f"Implement enhanced monitoring for {metric if metric else 'production metrics'}",
        "priority": "long-term",
        "expected_outcome": "Early detection of future issues"
    })
    
    return recommendations[:5]  # Limit to top 5 recommendations


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

    if diagnosis_result.get("method"):
        output.append(f"\nMethod: {diagnosis_result['method']}")

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

    print("Generating Rule-Based Diagnostic Recommendation...")
    diagnosis = synthesize_diagnosis_mock(
        entities=sample_entities,
        anomaly_findings=sample_anomalies,
        historical_incidents=sample_incidents
    )

    print(format_diagnosis(diagnosis))
    print("\n" + "=" * 80)
    print("RAW JSON OUTPUT")
    print("=" * 80)
    print(json.dumps(diagnosis, indent=2))
