import os
import sys
import json
import pandas as pd

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.drift_engine import FormattingDriftEngine

def run_full_matrix_analysis():
    print("=" * 80)
    print("      COMPLETE 2x2 MATRIX: QUANTIZATION DRIFT & SCALE ANALYSIS")
    print("=" * 80)

    runs = {
        "0.5B-FP16": "results/raw_outputs/eval_results_Qwen2.5-0.5B-FP16.jsonl",
        "0.5B-INT8": "results/raw_outputs/eval_results_Qwen2.5-0.5B-INT8.jsonl",
        "1.5B-FP16": "results/raw_outputs/eval_results_Qwen2.5-1.5B-FP16.jsonl",
        "1.5B-INT8": "results/raw_outputs/eval_results_Qwen2.5-1.5B-INT8.jsonl",
    }

    data = {}
    for tag, path in runs.items():
        if not os.path.exists(path):
            print(f"[-] Missing log file: {path}")
            return
        with open(path, "r", encoding="utf-8") as f:
            data[tag] = [json.loads(line) for line in f if line.strip()]

    drift_engine = FormattingDriftEngine(sample_size=100)

    summary_table = []
    
    pairs = [
        ("0.5B-FP16", "0.5B-INT8", "Qwen2.5-0.5B"),
        ("1.5B-FP16", "1.5B-INT8", "Qwen2.5-1.5B")
    ]

    for base_tag, quant_tag, model_label in pairs:
        df_base = pd.DataFrame(data[base_tag])
        df_quant = pd.DataFrame(data[quant_tag])

        p_raw_base = float(df_base["strict_raw_pass"].mean())
        p_scrub_base = float(df_base["scrubbed_pass"].mean())
        p_sem_base = float((df_base["parsed_output"] == df_base["ground_truth"]).mean())

        p_raw_quant = float(df_quant["strict_raw_pass"].mean())
        p_scrub_quant = float(df_quant["scrubbed_pass"].mean())
        p_sem_quant = float((df_quant["parsed_output"] == df_quant["ground_truth"]).mean())

        stats_res = drift_engine.calculate_drift_and_significance(
            p_base=p_scrub_base,
            p_quant=p_scrub_quant,
            semantic_base=p_sem_base,
            semantic_quant=p_sem_quant
        )

        summary_table.append({
            "Model": model_label,
            "FP16 Raw Pass (%)": p_raw_base * 100,
            "FP16 Scrubbed Pass (%)": p_scrub_base * 100,
            "FP16 Semantic Acc (%)": p_sem_base * 100,
            "INT8 Raw Pass (%)": p_raw_quant * 100,
            "INT8 Scrubbed Pass (%)": p_scrub_quant * 100,
            "INT8 Semantic Acc (%)": p_sem_quant * 100,
            "D_formatting (%)": stats_res["d_formatting_percent"],
            "Z-Score": stats_res["z_statistic"],
            "p-value": stats_res["p_value"],
            "Stat. Significant": stats_res["is_significant_p005"]
        })

    df_summary = pd.DataFrame(summary_table)
    print("\n" + df_summary.to_string(index=False))

    # Save final aggregate CSV
    os.makedirs("results", exist_ok=True)
    out_csv = "results/final_2x2_matrix_results.csv"
    df_summary.to_csv(out_csv, index=False)
    print(f"\n[+] Full comparative matrix report saved to: {out_csv}")

if __name__ == "__main__":
    run_full_matrix_analysis()