# Dataset and Baseline Validation Sheet

**Project Title:** Tracking Performance Drift in Large Language Models Across Successive Version Updates  
**Author:** Vedartha Sai Y (IIITA)  
**Selected Scope:** Quantifying Formatting Drift & Schema Compliance ($D_{\text{formatting}}$) in Quantized Local LLMs  
**Constraint Profile:** $0.00 API Token Cost | Local Execution (`Qwen-2.5-0.5B-Instruct` & `1.5B-Instruct`) | Consumer Hardware

---

## 1. Primary Dataset Specification

### JSON-Schema-Eval Micro-Suite
* **Official Source to Verify:** Synthesized and verified locally via open-source schema generation scripts derived from `JSON-Mode-Eval` and `Outlines` benchmark protocols.
* **File Structure & Schema (`json_schema_eval_100.jsonl`):**
  ```json
  {
    "id": "task_001",
    "tier": "nested_object",
    "input_context": "Extract user profile details from raw transcript...",
    "pydantic_schema_def": "class UserProfile(BaseModel): name: str; age: int",
    "target_json_schema": {
      "type": "object",
      "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},
      "required": ["name", "age"]
    }
  }
  ```
* **Benchmark Protocol:**
  1. Prompt model with `input_context` and JSON schema definition.
  2. Capture raw autoregressive completion string ($T = 0.0$, greedy decoding).
  3. Validate completion against `pydantic.BaseModel` schema definition.
  4. Record binary compliance pass score $P_{\text{format}} \in \{0, 1\}$.
* **Licensing & Access Terms:** Open Access / Creative Commons CC-BY 4.0.
* **Compute Requirement:** 100 prompts $\times$ ~200 generation tokens = ~20,000 tokens. Execution time: ~2–4 minutes per checkpoint on standard CPU/laptop hardware.
* **Common Pitfall:** Models wrapping outputs in markdown code fences (````json ... ````) when instructions specify raw JSON, triggering parsing exceptions unless handled via dual-pass validation.

---

## 2. Baseline Controls & Comparative Methods

### Baseline Method 1: Unconstrained Zero-Shot System Prompting (Naive Control)
* **Official Source to Verify:** Standard longitudinal API prompt protocol established by Chen et al. (2024) and Liang et al. (2022) (HELM).
* **Protocol & Purpose:** Evaluates raw model capacity to adhere to system instructions without runtime decoding constraints. Measures natural formatting drift under post-training weight quantization.
* **License & Compute:** MIT / Apache 2.0. Identical inference compute load to standard generation.
* **Pitfall:** Conversational preambles (e.g., *"Sure, here is your JSON:"*) inflate structural error rates if regex extraction is not isolated.

### Baseline Method 2: BNF Grammar-Constrained / Logit-Masked Decoding (Guided-Syntax Control)
* **Official Source to Verify:** `llama.cpp` GBNF (Grammar-Based Context-Free Parsing) or `Outlines` guided-decoding logit-masking engine.
* **Protocol & Purpose:** Dynamically masks invalid schema logits to $-\infty$ during sampling. Isolates whether quantization damages internal model reasoning versus syntactic output formatting.
* **License & Compute:** MIT License. Introduces ~5–10% CPU latency overhead per token due to dynamic grammar constraint evaluation.
* **Pitfall:** Sampling deadlocks: if low-bit quantization (INT4) suppresses valid logit probabilities below precision thresholds, grammar masking can trigger infinite loops or repeating token loops.

---

## 3. Evaluation Design & Stratified Partitioning

Because longitudinal drift and quantization evaluations assess pre-trained checkpoints under inference-only conditions, standard train/validation/test data splits are not applicable. Instead, the benchmark uses **Complexity-Stratified Partitioning**:

| Stratified Tier | Structural Characteristics | Sample Prompt Objective | Target Output Schema |
| :--- | :--- | :--- | :--- |
| **Tier 1: Primitive Flat** | 3–5 top-level key-value pairs using primitive types (`str`, `int`, `bool`, `enum`). | Extract user details (name, age, active status). | `{"name": str, "age": int, "is_active": bool}` |
| **Tier 2: Nested Objects** | 2–3 levels of nested child objects and required sub-fields. | Extract corporate structure and department heads. | `{"company": str, "department": {"head": str}}` |
| **Tier 3: Typed Arrays** | Variable-length array collections containing nested objects. | Extract medical prescription list with dosage. | `{"patient": str, "meds": [{"name": str, "mg": float}]}` |

---

## 4. Benchmark Leakage Risks & Mitigation Matrix

| Leakage Risk | Vulnerability Mechanism | Mitigation Strategy |
| :--- | :--- | :--- |
| **Pre-training Contamination** | Public JSON benchmarks present in pre-training corpora (e.g., Qwen-2.5 pre-training data). | Synthesize novel, dynamic extraction contexts using domain-specific synthetic schema keys. |
| **Schema Key Memorization** | Standard keys (`"name"`, `"age"`) memorized during instruction alignment, hiding formatting drops. | Include obscure field names (e.g., `telemetry_hz_rate`, `actuator_status_enum`) to force structural generation. |
| **Markdown Fence Leakage** | Prepend of ```json code wrappers across Instruct vs Base checkpoints. | Dual-pass parsing: record strict raw pass rate ($P_{\text{raw}}$) and scrubbed pass rate ($P_{\text{scrubbed}}$). |

---

## 5. Simplest Credible Baseline Setup

* **Configuration:** Unconstrained Zero-Shot Inference on `Qwen-2.5-0.5B-Instruct` across FP16 (Baseline) and INT4 (`Q4_K_M` GGUF) checkpoints evaluating 100 prompts.
* **Output Metrics:**
  * Schema Compliance Pass Rate ($P_{\text{format}}$)
  * Relative Formatting Drift ($D_{\text{formatting}} = \frac{P_{\text{INT4}} - P_{\text{FP16}}}{P_{\text{FP16}}} \times 100$)
  * Statistical Test Statistic ($Z = \frac{\bar{X}_{\text{INT4}} - \bar{X}_{\text{FP16}}}{\sqrt{\frac{s_{\text{FP16}}^2}{n} + \frac{s_{\text{INT4}}^2}{n}}}$)

---

## 6. Independent Pre-Execution Verification Audit

- [ ] **Model Checkpoint Verification:** Local existence of `Qwen-2.5-0.5B-Instruct` FP16 and INT4 (`Q4_K_M` GGUF) weights confirmed.
- [ ] **Deterministic Parameter Audit:** Inference settings enforced at `temperature = 0.0`, `top_p = 1.0`, `seed = 42`.
- [ ] **Pydantic Validation Isolation:** `pydantic.ValidationError` logged separately from `json.JSONDecodeError` to separate syntax from schema field errors.
- [ ] **Dual-Pass Parser Verification:** Evaluator records both raw pass rate ($P_{\text{raw}}$) and markdown-stripped pass rate ($P_{\text{scrubbed}}$).
- [ ] **Statistical Engine Audit:** Script correctly calculates two-proportion $Z$-score significance (Equation 6) and relative drift $D_{\text{formatting}}$ (Equation 1).
