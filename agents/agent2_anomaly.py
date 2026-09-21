"""
Agent 2: Anomaly Detection for Production Metrics
Computes z-scores to flag anomalies (values beyond 2 standard deviations)
and calculates correlations between metrics using pandas and scipy.
"""

import pandas as pd
import numpy as np
from scipy import stats
import json


def detect_anomalies(df: pd.DataFrame, metric_name: str, threshold: float = 2.0) -> dict:
    """
    Detect anomalies in a production metric using z-score method.
    
    Args:
        df: DataFrame containing production readings with metric columns
        metric_name: Name of the metric column to analyze
        threshold: Number of standard deviations for anomaly flagging (default: 2.0)
    
    Returns:
        JSON dict containing:
        - metric: Name of the analyzed metric
        - total_readings: Total number of data points
        - anomaly_count: Number of flagged anomalies
        - anomaly_percentage: Percentage of readings that are anomalies
        - anomalies: List of dicts with index, value, z_score for each anomaly
        - statistics: Mean, std, min, max of the metric
    
    Note: Z-score measures how many standard deviations a value is from the mean.
    Values with |z-score| > threshold are flagged as anomalies.
    """
    # Validate input
    if metric_name not in df.columns:
        raise ValueError(f"Metric '{metric_name}' not found in DataFrame columns: {df.columns.tolist()}")
    
    # Extract metric values
    metric_values = df[metric_name].dropna()
    
    if len(metric_values) == 0:
        return {
            "metric": metric_name,
            "error": "No valid data points for this metric"
        }
    
    # Compute z-scores using scipy.stats.zscore
    z_scores = stats.zscore(metric_values)
    
    # Flag anomalies: values beyond threshold standard deviations
    anomaly_mask = np.abs(z_scores) > threshold
    anomaly_indices = metric_values[anomaly_mask].index.tolist()
    anomaly_values = metric_values[anomaly_mask].tolist()
    anomaly_z_scores = z_scores[anomaly_mask].tolist()
    
    # Build anomalies list
    anomalies = [
        {
            "index": int(idx),
            "value": float(val),
            "z_score": float(z)
        }
        for idx, val, z in zip(anomaly_indices, anomaly_values, anomaly_z_scores)
    ]
    
    # Compute basic statistics
    statistics = {
        "mean": float(metric_values.mean()),
        "std": float(metric_values.std()),
        "min": float(metric_values.min()),
        "max": float(metric_values.max()),
        "count": int(len(metric_values))
    }
    
    return {
        "metric": metric_name,
        "threshold": threshold,
        "total_readings": int(len(metric_values)),
        "anomaly_count": len(anomalies),
        "anomaly_percentage": round(len(anomalies) / len(metric_values) * 100, 2),
        "anomalies": anomalies,
        "statistics": statistics
    }


def compute_correlations(df: pd.DataFrame, method: str = "pearson") -> dict:
    """
    Compute correlation matrix between all numeric metrics in the DataFrame.
    
    Args:
        df: DataFrame containing production metrics
        method: Correlation method - 'pearson', 'spearman', or 'kendall' (default: 'pearson')
    
    Returns:
        JSON dict containing:
        - method: Correlation method used
        - correlation_matrix: Dict of dicts with correlation coefficients
        - strong_correlations: List of metric pairs with |correlation| > 0.7
    
    Note: 
    - Pearson: Linear correlation (default, assumes normal distribution)
    - Spearman: Rank correlation (monotonic relationships, non-parametric)
    - Kendall: Ordinal association (robust to outliers)
    """
    # Select only numeric columns
    numeric_df = df.select_dtypes(include=[np.number])
    
    if numeric_df.empty:
        return {
            "method": method,
            "error": "No numeric columns found in DataFrame"
        }
    
    # Compute correlation matrix
    corr_matrix = numeric_df.corr(method=method)
    
    # Convert to nested dict format for JSON serialization
    correlation_dict = {}
    for col1 in corr_matrix.columns:
        correlation_dict[col1] = {}
        for col2 in corr_matrix.columns:
            correlation_dict[col1][col2] = round(corr_matrix.loc[col1, col2], 4)
    
    # Find strong correlations (|correlation| > 0.7)
    strong_correlations = []
    for i, col1 in enumerate(corr_matrix.columns):
        for j, col2 in enumerate(corr_matrix.columns):
            if i < j:  # Avoid duplicates and self-correlations
                corr_value = corr_matrix.loc[col1, col2]
                if abs(corr_value) > 0.7:
                    strong_correlations.append({
                        "metric_1": col1,
                        "metric_2": col2,
                        "correlation": round(corr_value, 4),
                        "strength": "strong_positive" if corr_value > 0 else "strong_negative"
                    })
    
    return {
        "method": method,
        "metrics": corr_matrix.columns.tolist(),
        "correlation_matrix": correlation_dict,
        "strong_correlations": strong_correlations
    }


def analyze_production_data(df: pd.DataFrame, metric_name: str = None, threshold: float = 2.0) -> dict:
    """
    Comprehensive analysis: detect anomalies for a specific metric and compute
    correlations between all metrics.
    
    Args:
        df: DataFrame containing production readings
        metric_name: Name of metric to analyze for anomalies (optional)
        threshold: Z-score threshold for anomaly detection (default: 2.0)
    
    Returns:
        JSON dict containing both anomaly detection and correlation analysis
    """
    result = {
        "correlation_analysis": compute_correlations(df)
    }
    
    if metric_name:
        result["anomaly_detection"] = detect_anomalies(df, metric_name, threshold)
    
    return result


# ── Example usage (uncomment to test) ───────────────────────────────────────────────
if __name__ == "__main__":
    # Create sample production data
    np.random.seed(42)
    sample_data = {
        "elongation": np.random.normal(100, 10, 100),
        "strength": np.random.normal(50, 5, 100),
        "tension": np.random.normal(30, 3, 100),
        "density": np.random.normal(200, 20, 100)
    }
    
    # Add some anomalies
    sample_data["elongation"][10] = 150  # High anomaly
    sample_data["elongation"][20] = 40   # Low anomaly
    sample_data["strength"][15] = 80     # High anomaly
    
    df = pd.DataFrame(sample_data)
    
    # Test anomaly detection
    print("=== Anomaly Detection for 'elongation' ===")
    anomalies = detect_anomalies(df, "elongation", threshold=2.0)
    print(json.dumps(anomalies, indent=2))
    
    print("\n=== Correlation Analysis ===")
    correlations = compute_correlations(df)
    print(json.dumps(correlations, indent=2))
    
    print("\n=== Comprehensive Analysis ===")
    full_analysis = analyze_production_data(df, metric_name="elongation")
    print(json.dumps(full_analysis, indent=2))
