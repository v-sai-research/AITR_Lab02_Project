import os
import sys
import json
import argparse
import pandas as pd
from typing import List, Dict, Any

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.dataset import BenchmarkDatasetLoader
from src.verifier import DualPassSchemaVerifier
from src.inference import LocalInferenceEngine
from src.drift_engine import FormattingDriftEngine

def run_pipeline(model_id: str, sample_size: int = 100, output_tag: str = "Qwen2.5-0.5B-FP16", quant_bits: int = None):
    os.makedirs("results/raw_outputs", exist_ok=True)
    
    print("=" * 70)
    print(f"      STARTING EVALUATION PIPELINE: {model_id} (Quant: {quant_bits}-bit, N={sample_size})")
    print("=" * 70)

    # 1. Load Dataset
    loader = BenchmarkDatasetLoader("data/json_schema_eval_100.jsonl")
    full_dataset = loader.load_and_validate()
    eval_subset = full_dataset[:sample_size]
    print(f"[+] Loaded {len(eval_subset)} evaluation items.")

    # 2. Initialize Inference Engine with Quantization Toggle
    engine = LocalInferenceEngine(model_id=model_id, device="cpu", quant_bits=quant_bits)
    engine.load_model()
    verifier = DualPassSchemaVerifier()

    # 3. Execute Deterministic Generation & Verification
    raw_results = []
    print("\n[+] Executing generation and verification pass...")
    
    for idx, item in enumerate(eval_subset, 1):
        prompt_id = item["id"]
        context = item["context"]
        schema = item["pydantic_schema"]
        ground_truth = item["ground_truth"]

        raw_completion = engine.generate_completion(context, schema)
        v_res = verifier.verify_completion(raw_completion, schema)
        
        record = {
            "id": prompt_id,
            "schema_tier": item["schema_tier"],
            "raw_completion": raw_completion,
            "strict_raw_pass": v_res["strict_raw_pass"],
            "scrubbed_pass": v_res["scrubbed_pass"],
            "pass_mode": v_res["pass_mode"],
            "error_category": v_res["error_category"],
            "parsed_output": v_res["parsed_output"],
            "ground_truth": ground_truth
        }
        raw_results.append(record)
        
        status_str = f"PASS ({v_res['pass_mode']})" if v_res['scrubbed_pass'] else f"FAIL ({v_res['error_category']})"
        print(f"    [{idx:03d}/{sample_size:03d}] Prompt ID {prompt_id:<18} -> {status_str}")

    # Save Raw Outputs
    raw_output_path = os.path.join("results", "raw_outputs", f"eval_results_{output_tag}.jsonl")
    with open(raw_output_path, "w", encoding="utf-8") as f:
        for r in raw_results:
            f.write(json.dumps(r) + "\n")
    print(f"\n[+] Raw generation logs saved to: {raw_output_path}")

    # 4. Compute Statistical Metrics
    drift_calc = FormattingDriftEngine(sample_size=sample_size)
    p_raw = drift_calc.compute_pass_rate(raw_results, pass_key="strict_raw_pass")
    p_scrubbed = drift_calc.compute_pass_rate(raw_results, pass_key="scrubbed_pass")
    p_semantic = drift_calc.compute_semantic_accuracy(raw_results, [item["ground_truth"] for item in eval_subset])

    summary = {
        "model_tag": output_tag,
        "sample_size": sample_size,
        "strict_raw_pass_rate": round(p_raw * 100, 2),
        "scrubbed_pass_rate": round(p_scrubbed * 100, 2),
        "semantic_accuracy": round(p_semantic * 100, 2)
    }

    print("\n" + "=" * 70)
    print("                  EVALUATION RUN SUMMARY")
    print("=" * 70)
    print(f" Model Tag          : {summary['model_tag']}")
    print(f" Evaluated Prompts  : {summary['sample_size']}")
    print(f" Strict Raw Pass    : {summary['strict_raw_pass_rate']}%")
    print(f" Scrubbed Pass      : {summary['scrubbed_pass_rate']}%")
    print(f" Semantic Accuracy  : {summary['semantic_accuracy']}%")
    print("=" * 70)

    # Append to Summary CSV
    csv_path = os.path.join("results", "summary_metrics.csv")
    df_summary = pd.DataFrame([summary])
    if not os.path.exists(csv_path):
        df_summary.to_csv(csv_path, index=False)
    else:
        df_summary.to_csv(csv_path, mode="a", header=False, index=False)
        
    print(f"[+] Summary metric logged to: {csv_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run LLM Formatting Drift Evaluation Pipeline")
    parser.add_argument("--model-id", type=str, default="Qwen/Qwen2.5-0.5B-Instruct", help="HuggingFace model ID")
    parser.add_argument("--sample-size", type=int, default=100, help="Number of benchmark samples to evaluate")
    parser.add_argument("--output-tag", type=str, default="Qwen2.5-0.5B-FP16", help="Output identifier tag")
    parser.add_argument("--quant-bits", type=int, default=None, help="Quantization bits (8 for INT8 dynamic)")
    
    args = parser.parse_args()
    run_pipeline(args.model_id, sample_size=args.sample_size, output_tag=args.output_tag, quant_bits=args.quant_bits)