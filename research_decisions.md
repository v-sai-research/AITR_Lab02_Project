# Research Decisions, Literature Mapping, and Protocol Rationale

**Project:** Empirical Investigation of Formatting Drift ($D_{\text{formatting}}$) in Quantized Small Language Models  
**Author:** Vedartha Sai Y (IIITA)  
**Document Status:** Finalized / Publication-Ready

---

## 1. Executive Summary & Core Research Questions

This document records the foundational research decisions, literature motivations, experimental controls, and analytical interpretations behind the **LLM Formatting Drift** evaluation framework.

The study investigates how post-training weight quantization interacts with parameter scale to affect structured JSON schema adherence across three formal research questions:

* **$RQ_1$ (Schema Complexity Nonlinearity):** Is formatting drift ($D_{\text{formatting}}$) non-linearly amplified as target schema complexity increases from flat primitive key-value pairs to nested hierarchies and array sequences?
* **$RQ_2$ (Parameter Capacity Buffering Hypothesis):** Does scaling parameter capacity ($0.5\text{B} \rightarrow 1.5\text{B}$) buffer against post-training quantization formatting drift under identical execution parameters?
* **$RQ_3$ (Surface vs. Structural Degradation):** What proportion of quantization drift stems from surface markdown code fence additions ($P_{\text{raw}} \rightarrow P_{\text{scrubbed}}$) versus inner schema structural corruption (`SYNTAX_ERROR` / `SCHEMA_FIELD_MISMATCH`)?

---

## 2. Literature-to-Experiment Mapping Matrix

| Theoretical Concept / Literature | Key Reference(s) | Experimental Implementation in Study | Protocol / Analytical Rationale |
| :--- | :--- | :--- | :--- |
| **8-Bit Matrix Multiplication & Outlier Activations** | Dettmers et al. (*LLM.int8()*, NeurIPS 2022) | Evaluated dynamic linear quantization (`torch.ao.quantization.quantize_dynamic`) across transformer projection layers. | Isolates the impact of uncalibrated per-tensor INT8 quantization on token-level attention logit distributions. |
| **Post-Training Quantization (PTQ) Noise** | Frantar et al. (*GPTQ*, ICLR 2023); Lin et al. (*AWQ*, MLSys 2024) | Compared unquantized $\text{FP16}$ base checkpoints against dynamic $\text{INT8}$ quantized variants. | Establishes the baseline structural capacity ($P_{\text{base}} = 100.0\%$) to calculate exact relative drift ($D_{\text{formatting}}$). |
| **Activation Smoothing & Quantization Degradation** | Xiao et al. (*SmoothQuant*, ICML 2023) | Tracked token generation failure modes (`SYNTAX_ERROR`, `SCHEMA_FIELD_MISMATCH`). | Explains why uncalibrated dynamic quantization clips activation outliers in self-attention heads, causing *attention routing collapse*. |
| **Grammar-Constrained LLM Decoding** | Willard & Louf (*Outlines*, 2023) | Evaluated models under **unconstrained** greedy decoding without external logit masking. | Measures the *intrinsic resilience* of fine-tuned model weights to maintain schema constraints without external runtime masking. |
| **JSON Schema Structural Validation** | Wright & Andrews (*Draft-07 JSON Schema*, IETF 2020) | Built a dual-pass schema validation engine using Python `jsonschema` and `pydantic`. | Replaces subjective LLM-as-a-judge scoring with deterministic, boolean schema compliance checks. |

---

## 3. Methodological Rationale & Experimental Controls

### 3.1 Parameter Scale Selection ($0.5\text{B}$ vs. $1.5\text{B}$)
* **Decision:** Selected `Qwen2.5-0.5B-Instruct` ($0.49\text{B}$ parameters) and `Qwen2.5-1.5B-Instruct` ($1.54\text{B}$ parameters).
* **Rationale:** Sub-2B parameter models represent the target class for edge/local CPU deployments. Comparing $0.5\text{B}$ against $1.5\text{B}$ within the same model architecture family (`Qwen2.5-Instruct`) controls for tokenizer vocabulary, pre-training corpus, and instruction-tuning methodology while testing a $3.14\times$ parameter scale increase.

### 3.2 Deterministic Decoding Protocol
* **Decision:** Fixed sampling hyperparameters to greedy decoding:
  * Temperature ($T = 0.0$), Top-$p = 1.0$, `do_sample = False`.
  * Random Seed: Locked at $\text{seed} = 42$ across PyTorch, NumPy, and Python.
* **Rationale:** Eliminates stochastic variance across evaluation runs, guaranteeing that observed pass rate deltas stem strictly from weight precision reduction rather than sampling entropy.

### 3.3 Dual-Pass Verification Logic
* **Decision:** Implemented a two-pass compliance engine:
  * **Pass 1 ($P_{\text{raw}}$):** Direct JSON parse on unedited raw output stream.
  * **Pass 2 ($P_{\text{scrubbed}}$):** Regex extraction of JSON blocks enclosed within markdown code fences (` ```json ... ``` `).
* **Rationale:** Disambiguates surface-level formatting violations (adding markdown code block wrappers) from severe structural corruption (unparseable JSON syntax or missing schema fields).

### 3.4 Stratified Complexity Tiers
* **Decision:** Partitioned the $N=100$ evaluation prompts into three difficulty tiers:
  * **Tier 1 (Primitive Flat, $N_1=35$):** Single-level objects containing primitive data types (`string`, `integer`, `boolean`).
  * **Tier 2 (Nested Object, $N_2=35$):** Multi-level object hierarchies requiring recursive depth tracking.
  * **Tier 3 (Typed Array, $N_3=30$):** Array structures requiring sequential element type preservation.
* **Rationale:** Tests whether schema compliance degradation scales non-linearly with recursive structural complexity.

---

## 4. Hypothesis Testing & Empirical Verdicts

| Formulated Hypothesis | Tested Empirical Condition | Observed Result | Statistical Evidence | Final Verdict |
| :--- | :--- | :--- | :--- | :--- |
| **$H_1$: Post-training quantization induces statistically significant formatting drift ($D_{\text{formatting}} < 0$).** | $\text{FP16} \rightarrow \text{INT8}$ pass rate comparison across 100 prompts. | Pass rate dropped from $100.0\%$ to $7.0\%$ ($0.5\text{B}$) and $0.0\%$ ($1.5\text{B}$). | $Z = -13.1845$ ($0.5\text{B}$), $Z = -14.1421$ ($1.5\text{B}$), $p < 10^{-15}$. | **SUPPORTED** |
| **$H_2$: Formatting failure accelerates non-linearly with schema complexity.** | Tier 1 vs. Tier 2 vs. Tier 3 pass rates under $\text{INT8}$. | $0.5\text{B-INT8}$ achieved $20.0\%$ pass on Tier 1, but dropped to $0.0\%$ on Tiers 2 & 3. | Tier 1 $\Delta = -80.0\%$, Tier 2 $\Delta = -100.0\%$, Tier 3 $\Delta = -100.0\%$. | **SUPPORTED** |
| **$H_3$: Parameter capacity scaling ($0.5\text{B} \rightarrow 1.5\text{B}$) buffers against formatting drift.** | $0.5\text{B-INT8}$ vs. $1.5\text{B-INT8}$ scrubbed pass rates. | $1.5\text{B-INT8}$ suffered $0.0\%$ pass rate, performing worse than $0.5\text{B-INT8}$ ($7.0\%$). | $D_{\text{formatting}} = -93.0\%$ ($0.5\text{B}$) vs. $D_{\text{formatting}} = -100.0\%$ ($1.5\text{B}$). | **CONTRADICTED (REFUTED)** |
| **$H_4$: Surface regex scrubbing eliminates quantization-induced failures.** | $P_{\text{raw}}$ vs. $P_{\text{scrubbed}}$ under $\text{INT8}$ precision. | Scrubbing recovered only $7.0\%$ of $0.5\text{B-INT8}$ runs and $0.0\%$ of $1.5\text{B-INT8}$ runs. | $P_{\text{raw}} = 0.0\%$ for both models; $P_{\text{scrubbed}} = 7.0\%$ ($0.5\text{B}$) and $0.0\%$ ($1.5\text{B}$). | **CONTRADICTED (REFUTED)** |

---

## 5. Failure Mechanics & Diagnostic Interpretations

Categorization of $N=193$ failure outputs across dynamic $\text{INT8}$ runs identified two primary failure mechanics:

```
+-----------------------------------------------------------------------------------+
|                        INT8 FAILURE TAXONOMY DISTRIBUTION                         |
+-----------------------------------------------------------------------------------+
|  [ SYNTAX_ERROR ]                                                                 |
|  ██████████████████████████████████████████████████████████ 67.9% (131/193)       |
|  • Preambles ("Assistant:"), markdown bullet lists, stray HTML tags (</pre>)      |
|                                                                                   |
|  [ SCHEMA_FIELD_MISMATCH ]                                                        |
|  ██████████████████████████ 31.6% (61/193)                                        |
|  • Schema definition echoing: Model re-describes the input JSON Schema template  |
|                                                                                   |
|  [ SCHEMA_TYPE_MISMATCH ]                                                         |
|  █ 0.5% (1/193)                                                                   |
|  • Valid JSON instance generated, but field value type deviates from ground truth |
+-----------------------------------------------------------------------------------+
```

### 5.1 System Prompt Role Erosion ($67.9\%$ of failures)
Quantization noise distorts the logit deltas between structural tokens (`{`, `"`) and conversational stop/transition tokens. As a result, the model ignores the frozen system prompt (*"Output ONLY valid JSON"*), allowing chat role artifacts (`Assistant:`), preambles (`Question:... Answer:...`), and HTML tags (`</pre>`) to leak into the generation stream.

### 5.2 Attention Routing Collapse & Schema Echoing ($31.6\%$ of failures)
Instead of populating key-value instances with context values, quantized models treat the prompt's input JSON Schema as a template to repeat back. Uncalibrated quantization degrades the self-attention heads responsible for mapping contextual variables to target fields, causing the model to default to repeating high-frequency structural patterns present in the prompt (the schema definition itself).

---

## 6. Boundary Scoping & Methodological Limitations

To prevent over-generalization, the conclusions of this study are bound by the following explicit constraints:

1. **Quantization Scope:** Findings apply specifically to **uncalibrated PyTorch dynamic CPU linear quantization** (`torch.ao.quantization.quantize_dynamic`). Results must **not** be extrapolated to static block-wise quantization (e.g., GGUF `Q8_0`, `Q4_K_M`) or calibration-based post-training quantization (GPTQ, AWQ).
2. **Unconstrained Generation Scope:** This evaluation tested *intrinsic weight adherence*. In production environments, grammar-constrained decoding engines (e.g., Outlines, vLLM guided decoding) can forcibly mask invalid tokens at runtime, providing structural guarantees regardless of weight quantization noise.
3. **Model Scale & Architecture Scope:** Experiments were executed on sub-2B parameter instruction-tuned models (`Qwen2.5-Instruct`). Scaling trends may differ for larger parameter architectures ($7\text{B}, 70\text{B}$).