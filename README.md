# Empirical Investigation of LLM Formatting Drift ($D_{\text{formatting}}$)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

This repository contains the dataset, modular Python benchmarking stack, and experimental analysis scripts for reproducing the study **"Tracking Performance Drift in Large Language Models Across Successive Version Updates: Empirical Investigation of Parameter Scale and Post-Training Quantization Dynamics"**.

---

## 1. Problem Statement

When Large Language Models (LLMs) are deployed in production agentic pipelines, downstream software modules require outputs that adhere strictly to formal specifications (e.g., Draft-07 JSON Schema). While model compression techniques like post-training quantization lower RAM footprints and execution latency, they introduce **Formatting Drift ($D_{\text{formatting}}$)**—a severe, statistically significant regression in structured schema adherence.

This project empirically measures formatting drift across a $2 \times 2$ factorial evaluation matrix:
* **Model Scale:** `Qwen2.5-0.5B-Instruct` vs. `Qwen2.5-1.5B-Instruct`
* **Precision:** Unquantized $\text{FP16}$ vs. Dynamic $\text{INT8}$ (`torch.ao.quantization`)
* **Evaluation Target:** $N=100$ stratified prompt schemas across 3 complexity tiers under deterministic greedy decoding ($T=0.0$, $\text{seed}=42$).

---

## 2. Environment & Hardware Specifications

### Verified Execution Environment
* **OS:** Microsoft Windows 10 / 11 (64-bit)
* **CPU Execution:** System RAM: $15.44 \text{ GB}$ (No dedicated GPU required)
* **Python Runtime:** Python `3.10.x` or `3.11.x` inside a local virtual environment (`venv`)

### Requirements & Key Dependencies (`requirements.txt`)
```text
torch>=2.0.0
transformers>=4.38.0
pydantic>=2.0.0
jsonschema>=4.18.0
pandas>=2.0.0
scipy>=1.10.0
numpy>=1.24.0
```

---

## 3. Repository Structure

```text
llm_formatting_drift/
├── data/
│   └── json_schema_eval_100.jsonl      # N=100 stratified benchmark prompts
├── src/
│   ├── __init__.py
│   ├── dataset.py                      # Benchmark dataset loader & schema validator
│   ├── verifier.py                     # Dual-pass Pydantic/JSONSchema verifier
│   ├── inference.py                    # Deterministic CPU generation engine
│   └── drift_engine.py                 # Statistical Z-test & D_formatting calculator
├── scripts/
│   ├── run_evaluation.py               # Main pipeline orchestration script
│   ├── analyze_results.py              # Single-run stratified tier analyzer
│   ├── analyze_all_runs.py             # 2x2 matrix comparative aggregator
│   └── inspect_failures.py            # Qualitative failure diagnostic viewer
├── tests/
│   └── test_drift_engine.py            # Unit test suite for statistical math
├── results/
│   ├── raw_outputs/                    # Raw per-prompt JSONL generation logs
│   ├── summary_metrics.csv             # Execution log append table
│   ├── final_2x2_matrix_results.csv   # Aggregated 2x2 statistical matrix
│   └── drift_stratified_report.csv    # Tier-wise failure distribution report
├── README.md                           # Project documentation
└── requirements.txt                    # Python dependency lockfile
```

---

## 4. Setup & Installation

1. **Clone the Repository:**
   ```cmd
   git clone https://github.com/your-username/llm_formatting_drift.git
   cd llm_formatting_drift
   ```

2. **Create and Activate Virtual Environment:**
   * **Windows (CMD):**
     ```cmd
     python -m venv venv
     venv\Scripts\activate
     ```
   * **Linux / macOS:**
     ```bash
     python3 -m venv venv
     source venv/bin/activate
     ```

3. **Install Dependencies:**
   ```cmd
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Verify Environment Setup:**
   Run the unit test suite to verify the statistical drift calculations:
   ```cmd
   python tests\test_drift_engine.py
   ```

---

## 5. Dataset & Licensing

* **Dataset Path:** `data/json_schema_eval_100.jsonl`
* **Size:** 100 ground-truth context/schema prompt records.
* **Stratification Tiers:**
  * **Tier 1 (Primitive Flat, $N_1=35$):** Flat key-value pairs (`string`, `integer`, `boolean`).
  * **Tier 2 (Nested Object, $N_2=35$):** Multi-level object hierarchies with nested properties.
  * **Tier 3 (Typed Array, $N_3=30$):** Homogeneous and heterogeneous array structures.
* **License & Weight Access:** The evaluation prompts are original synthetic benchmarks provided under the MIT License. Model weights for `Qwen2.5-0.5B-Instruct` and `Qwen2.5-1.5B-Instruct` are fetched automatically from Hugging Face Hub under the permissive Qwen Research License. No proprietary keys or restricted weights are embedded in this repository.

---

## 6. Execution Protocol & Benchmark Commands

### 6.1 Genuinely Executed Pipeline Commands

Run all 4 experimental cells sequentially to generate the primary evaluation logs in `results/raw_outputs/`:

#### Cell 1: Baseline 0.5B Model (FP16)
```cmd
python scripts\run_evaluation.py --model-id Qwen/Qwen2.5-0.5B-Instruct --sample-size 100 --output-tag Qwen2.5-0.5B-FP16
```

#### Cell 2: Quantized 0.5B Model (Dynamic INT8)
```cmd
python scripts\run_evaluation.py --model-id Qwen/Qwen2.5-0.5B-Instruct --sample-size 100 --output-tag Qwen2.5-0.5B-INT8 --quant-bits 8
```

#### Cell 3: Baseline 1.5B Model (FP16)
```cmd
python scripts\run_evaluation.py --model-id Qwen/Qwen2.5-1.5B-Instruct --sample-size 100 --output-tag Qwen2.5-1.5B-FP16
```

#### Cell 4: Quantized 1.5B Model (Dynamic INT8)
```cmd
python scripts\run_evaluation.py --model-id Qwen/Qwen2.5-1.5B-Instruct --sample-size 100 --output-tag Qwen2.5-1.5B-INT8 --quant-bits 8
```

---

### 6.2 Evaluation & Analysis Commands

1. **Generate Full 2x2 Matrix Report:**
   ```cmd
   python scripts\analyze_all_runs.py
   ```
   *Output generated:* `results/final_2x2_matrix_results.csv`

2. **Generate Stratified Tier Breakdown (0.5B FP16 vs INT8):**
   ```cmd
   python scripts\analyze_results.py
   ```
   *Output generated:* `results/drift_stratified_report.csv`

3. **Run Qualitative Failure Diagnostics:**
   ```cmd
   python scripts\inspect_failures.py
   ```

---

### 6.3 Optional / Unexecuted Extension Commands `[TEST REQUIRED]`

The following commands represent prospective extensions (e.g., CUDA GPU execution or GGUF static block-wise testing) and have **not** been executed in the core Windows CPU baseline:

* **GPU CUDA Pipeline Execution `[TEST REQUIRED]`:**
  ```bash
  python scripts/run_evaluation.py --model-id Qwen/Qwen2.5-1.5B-Instruct --sample-size 100 --output-tag Qwen2.5-1.5B-CUDA --device cuda
  ```

* **GGUF Static Quantization Benchmark `[TEST REQUIRED]`:**
  ```bash
  python scripts/run_gguf_evaluation.py --model-path models/qwen2.5-1.5b-instruct-q4_k_m.gguf --sample-size 100
  ```

---

## 7. Measured Benchmark Results

### Consolidated 2x2 Factorial Matrix (`results/final_2x2_matrix_results.csv`)

| Model | FP16 Raw Pass | FP16 Scrubbed Pass | FP16 Semantic Acc | INT8 Raw Pass | INT8 Scrubbed Pass | INT8 Semantic Acc | Relative Drift ($D_{\text{fmt}}$) | $Z$-Statistic | $p$-value | Stat. Significant |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Qwen2.5-0.5B** | $100.0\%$ | $100.0\%$ | $94.0\%$ | $0.0\%$ | $7.0\%$ | $0.0\%$ | **$-93.0\%$** | -13.1845 | $< 10^{-15}$ | **True** |
| **Qwen2.5-1.5B** | $100.0\%$ | $100.0\%$ | $100.0\%$ | $0.0\%$ | $0.0\%$ | $0.0\%$ | **$-100.0\%$** | -14.1421 | $< 10^{-15}$ | **True** |

### Stratified Pass Rate by Complexity Tier (`results/drift_stratified_report.csv`)

| Schema Tier | Sample Count ($N$) | FP16 Pass Rate | 0.5B-INT8 Pass Rate | 1.5B-INT8 Pass Rate |
| :--- | :---: | :---: | :---: | :---: |
| **Tier 1: Primitive Flat** | 35 | $100.0\%$ | $20.0\%$ | $0.0\%$ |
| **Tier 2: Nested Object** | 35 | $100.0\%$ | $0.0\%$ | $0.0\%$ |
| **Tier 3: Typed Array** | 30 | $100.0\%$ | $0.0\%$ | $0.0\%$ |

---

## 8. Compute Requirements & Execution Time

| Execution Mode | Model Target | Peak System Memory (RAM) | Execution Time (100 Prompts) |
| :--- | :--- | :---: | :---: |
| Unquantized FP16 | `Qwen2.5-0.5B-Instruct` | $\sim 1.2 \text{ GB}$ | $\sim 12 \text{ minutes}$ |
| Dynamic INT8 | `Qwen2.5-0.5B-Instruct` | $\sim 0.7 \text{ GB}$ | $\sim 8 \text{ minutes}$ |
| Unquantized FP16 | `Qwen2.5-1.5B-Instruct` | $\sim 3.2 \text{ GB}$ | $\sim 28 \text{ minutes}$ |
| Dynamic INT8 | `Qwen2.5-1.5B-Instruct` | $\sim 1.8 \text{ GB}$ | $\sim 18 \text{ minutes}$ |

---

## 9. Methodological Limitations

1. **Quantization Scope:** This benchmark specifically evaluates PyTorch post-training dynamic per-tensor CPU quantization (`torch.ao.quantization.quantize_dynamic`). Results should not be generalized to static block-wise formats (e.g., GGUF `Q4_K_M`) or calibration-based PTQ (GPTQ, AWQ).
2. **Model Architecture Scope:** Experiments were conducted on sub-2B parameter checkpoints of `Qwen2.5-Instruct`.
3. **Unconstrained Inference:** Model outputs were generated without external grammar-constrained decoding (e.g., Outlines, JSONLogitsProcessor) to measure intrinsic weight resilience.

---

## 10. Minimum Reproduction Checklist for Independent Verification

To independently verify the empirical results of this study, follow this checklist:

- [ ] **Step 1:** Confirm system has $\ge 8\text{ GB}$ RAM available and Python 3.10+ installed.
- [ ] **Step 2:** Clone repo and set up a clean `venv`. Install `requirements.txt`.
- [ ] **Step 3:** Run unit test suite: `python tests\test_drift_engine.py`. Ensure all math assertions pass.
- [ ] **Step 4:** Execute Cell 1 (`Qwen2.5-0.5B-FP16`) and Cell 2 (`Qwen2.5-0.5B-INT8`).
- [ ] **Step 5:** Execute Cell 3 (`Qwen2.5-1.5B-FP16`) and Cell 4 (`Qwen2.5-1.5B-INT8`).
- [ ] **Step 6:** Run `python scripts\analyze_all_runs.py` and verify that `results/final_2x2_matrix_results.csv` reports:
  - 0.5B FP16 Scrubbed Pass = $100.0\%$
  - 0.5B INT8 Scrubbed Pass = $7.0\%$ ($D_{\text{fmt}} = -93.0\%$)
  - 1.5B FP16 Scrubbed Pass = $100.0\%$
  - 1.5B INT8 Scrubbed Pass = $0.0\%$ ($D_{\text{fmt}} = -100.0\%$)

---

## 11. Credits, License, & Citation

### License
This project is licensed under the **MIT License**. See `LICENSE` for details.

### Citation
If you utilize this benchmark pipeline or data in your research, please cite:

```bibtex
@techreport{vedartha2026formattingdrift,
  author       = {Vedartha Sai Y},
  title        = {Empirical Investigation of Formatting Drift: Parameter Scale and Post-Training Quantization Dynamics in Small Language Models},
  institution  = {Indian Institute of Information Technology, Allahabad (IIITA)},
  year         = {2026},
  type         = {Technical Research Report}
}
```