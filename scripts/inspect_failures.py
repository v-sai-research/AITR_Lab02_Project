import os
import sys
import json
import pandas as pd

def inspect_model_failures(log_path: str, model_tag: str, num_samples_per_category: int = 2):
    if not os.path.exists(log_path):
        print(f"[-] Log file not found: {log_path}")
        return

    print("=" * 80)
    print(f"      QUALITATIVE FAILURE DIAGNOSTICS: {model_tag}")
    print("=" * 80)

    with open(log_path, "r", encoding="utf-8") as f:
        records = [json.loads(line) for line in f if line.strip()]

    df = pd.DataFrame(records)
    failed_df = df[df["scrubbed_pass"] == False]

    print(f"[+] Total Evaluated Prompts : {len(df)}")
    print(f"[+] Total Failed Prompts    : {len(failed_df)}")
    print(f"[+] Failure Rate            : {(len(failed_df) / len(df)) * 100:.1f}%\n")

    categories = failed_df["error_category"].unique()

    for category in categories:
        print("-" * 80)
        print(f" ERROR CATEGORY: {category}")
        print("-" * 80)
        
        cat_samples = failed_df[failed_df["error_category"] == category].head(num_samples_per_category)
        
        for sample_idx, (_, row) in enumerate(cat_samples.iterrows(), 1):
            print(f" Sample {sample_idx} | Prompt ID: {row['id']} | Tier: {row['schema_tier']}")
            print(" Raw Completion Output:")
            print(" " + "-" * 70)
            # Display first 350 characters of raw completion for inspection
            raw_text = row["raw_completion"]
            preview_text = raw_text if len(raw_text) <= 350 else raw_text[:350] + "... [TRUNCATED]"
            print(f" {repr(preview_text)}")
            print(" " + "-" * 70 + "\n")

if __name__ == "__main__":
    inspect_model_failures("results/raw_outputs/eval_results_Qwen2.5-0.5B-INT8.jsonl", "Qwen2.5-0.5B-INT8")
    inspect_model_failures("results/raw_outputs/eval_results_Qwen2.5-1.5B-INT8.jsonl", "Qwen2.5-1.5B-INT8")