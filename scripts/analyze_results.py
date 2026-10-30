import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.drift_engine import FormattingDriftEngine

def run_comparative_analysis(fp16_path: str, int8_path: str):
    print("=" * 70)
    print("      STRATIFIED DRIFT & ERROR TAXONOMY ANALYSIS")
    print("=" * 70)

    # Read Raw Output Logs
    with open(fp16_path, "r", encoding="utf-8") as f:
        fp16_records = [json.loads(line) for line in f if line.strip()]

    with open(int8_path, "r", encoding="utf-8") as f:
        int8_records = [json.loads(line) for line in f if line.strip()]

    df_fp16 = pd.DataFrame(fp16_records)
    df_int8 = pd.DataFrame(int8_records)

    # 1. Compute Overall Drift & Z-Scores
    drift_engine = FormattingDriftEngine(sample_size=len(df_fp16))
    
    p_base_scrubbed = float(df_fp16["scrubbed_pass"].mean())
    p_quant_scrubbed = float(df_int8["scrubbed_pass"].mean())
    
    p_base_semantic = float((df_fp16["parsed_output"] == df_fp16["ground_truth"]).mean())
    p_quant_semantic = float((df_int8["parsed_output"] == df_int8["ground_truth"]).mean())

    stats_res = drift_engine.calculate_drift_and_significance(
        p_base=p_base_scrubbed,
        p_quant=p_quant_scrubbed,
        semantic_base=p_base_semantic,
        semantic_quant=p_quant_semantic
    )

    print(f"[+] Scrubbed Pass Rate FP16 : {p_base_scrubbed * 100:.1f}%")
    print(f"[+] Scrubbed Pass Rate INT8 : {p_quant_scrubbed * 100:.1f}%")
    print(f"[+] Relative Drift (D_fmt) : {stats_res['d_formatting_percent']}%")
    print(f"[+] Z-Statistic             : {stats_res['z_statistic']}")
    print(f"[+] p-value                 : {stats_res['p_value']}")
    print(f"[+] Significant (p < 0.05)   : {stats_res['is_significant_p005']}")

    # 2. Stratified Tier Breakdown
    print("\n" + "-" * 70)
    print("                 STRATIFIED PASS RATES BY SCHEMA TIER")
    print("-" * 70)
    
    tiers = df_fp16["schema_tier"].unique()
    tier_summary = []
    
    for t in sorted(tiers):
        sub_fp16 = df_fp16[df_fp16["schema_tier"] == t]
        sub_int8 = df_int8[df_int8["schema_tier"] == t]
        
        pass_fp16 = sub_fp16["scrubbed_pass"].mean() * 100
        pass_int8 = sub_int8["scrubbed_pass"].mean() * 100
        delta = pass_int8 - pass_fp16
        
        tier_summary.append({
            "Schema Tier": t,
            "Count": len(sub_fp16),
            "FP16 Pass (%)": round(pass_fp16, 1),
            "INT8 Pass (%)": round(pass_int8, 1),
            "Delta (%)": round(delta, 1)
        })
        print(f"  {t:<25} | FP16: {pass_fp16:5.1f}% | INT8: {pass_int8:5.1f}% | Delta: {delta:5.1f}%")

    # 3. Error Taxonomy Breakdown for INT8
    print("\n" + "-" * 70)
    print("                   INT8 ERROR TAXONOMY DISTRIBUTION")
    print("-" * 70)
    
    error_counts = df_int8[df_int8["scrubbed_pass"] == False]["error_category"].value_counts()
    for cat, count in error_counts.items():
        pct = (count / len(df_int8)) * 100
        print(f"  {cat:<25} : {count:3d} errors ({pct:5.1f}% of total dataset)")

    # Save CSV Report
    os.makedirs("results", exist_ok=True)
    report_path = "results/drift_stratified_report.csv"
    pd.DataFrame(tier_summary).to_csv(report_path, index=False)
    print(f"\n[+] Detailed stratified report saved to: {report_path}")

if __name__ == "__main__":
    fp16_log = "results/raw_outputs/eval_results_Qwen2.5-0.5B-FP16.jsonl"
    int8_log = "results/raw_outputs/eval_results_Qwen2.5-0.5B-INT8.jsonl"
    run_comparative_analysis(fp16_log, int8_log)