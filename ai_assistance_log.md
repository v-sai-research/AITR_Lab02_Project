# AI Assistance Log and Research Integrity Attestation

**Project Title:** Empirical Investigation of Formatting Drift ($D_{\text{formatting}}$) in Quantized Small Language Models  
**Author:** Vedartha Sai Y (IIITA)  
**Document Status:** Complete / Verified  

---

## 1. AI Provenance & Task Audit Matrix

The matrix below logs all significant AI-assisted activities performed across the research lifecycle, detailing the tool, initial proposal, human oversight, verification method, and final code/result diffs.

| Research Phase & Task | AI Tool & Version | Direct Input Prompt / Command | AI Initial Proposal | Human Decision & Critical Assessment | Verification Method | Code / Document Diff | Final Outcome |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Statistical Engine Design** (`src/drift_engine.py`) | Gemini 3 Flash (Web) | *"Implement statistical engine for two-proportion Z-test and relative drift metric."* | Proposed standard $Z$-test math without handling $P_{\text{pool}} = 0$ or $1.0$. | **Modified.** Division-by-zero crashes occur when pass rates drop to $0.0\%$. | Executed unit test suite (`tests/test_drift_engine.py`) with zero-pass edge cases. | Added numerical floor logic (`se = 1e-6` when $P_{\text{pool}} \in \{0, 1\}$). | Robust statistical module handling zero-pass conditions without crashes. |
| **Qualitative Diagnostics** (`scripts/inspect_failures.py`) | Gemini 3 Flash (Web) | *"Analyze 1.5B INT8 run outputs showing 0.0% pass rate."* | Suggested declaring complete model breakdown and drafting the final manuscript immediately. | **Rejected.** Premature to publish without investigating *why* generation failed at token level. | Executed qualitative parser on raw output streams (`eval_results_Qwen2.5-1.5B-INT8.jsonl`). | Implemented diagnostic script categorizing failure types (`SYNTAX_ERROR`, `SCHEMA_FIELD_MISMATCH`). | Identified two distinct breakdown mechanics: *Schema Echoing* ($31.6\%$) and *Role Erosion* ($67.9\%$). |
| **Manuscript Scoping & Scoping Review** | Gemini 3 Flash (Web) | *"Review draft manuscript as a critical technical reviewer."* | Drafted review claiming *all* post-training quantization destroys LLM schema compliance and refutes scaling laws universally. | **Corrected.** AI over-generalized findings from PyTorch dynamic CPU quantization to all quantization methods. | Literature audit of static block-wise PTQ (GPTQ, AWQ, GGUF). | Updated `formatting_drift_paper.tex` and `README.md` to explicitly scope claims to *uncalibrated dynamic CPU quantization*. | Scientifically defensible manuscript avoiding false claims regarding calibrated static quantization. |
| **BibTeX & Citation Formatting** | Gemini 3 Flash (Web) | *"Generate IEEE paper draft with citations."* | Generated paper draft with 5 general citations and missing core quantization references. | **Expanded.** Demanded inclusion of foundational PTQ references (GPTQ, AWQ, LLM.int8()). | Audited IEEE/ACM conference proceedings for exact citation metadata. | Added 3 formal references to `formatting_drift_paper.tex` (`\cite{gptq}`, `\cite{awq}`, `\cite{llm_int8}`). | Complete 8-reference IEEE bibliography matching conference publication requirements. |

---

## 2. Real Examples of AI Limitations and Manual Verification

### Example 1: Premature Synthesis and Process Skipping
* **AI Limitation:** Upon receiving $0.0\%$ pass rate logs for `Qwen2.5-1.5B-INT8`, the AI attempted to jump directly to drafting the paper summary, skipping qualitative output analysis.
* **Human Intervention:** The user halted the automated output (*"what are you planning next, dont do it, just tell"*) and enforced execution of qualitative diagnostics.
* **Resulting Insight:** Qualitative inspection revealed that $31.6\%$ of failures were **Schema Definition Echoing** (the model repeating schema structure definitions rather than extracting context data into instances). Bypassing this step would have left the research without its core failure taxonomy.

### Example 2: Over-Generalization of Negative Findings
* **AI Limitation:** The AI generated conclusions asserting that *"parameter scale fails to buffer against quantization across all LLMs"* and that *"INT8 quantization renders models unusable for JSON tasks."*
* **Human Intervention:** The human author recognized that the setup evaluated a specific uncalibrated implementation (`torch.ao.quantization.quantize_dynamic`) on CPU without activation calibration or logit masking.
* **Resulting Insight:** Re-scoped all paper claims, peer-review responses, and repository documentation to specify *uncalibrated dynamic per-tensor linear quantization*, preserving scientific rigor.

### Example 3: Edge-Case Handling in Statistical Formulas
* **AI Limitation:** Initial AI code in `src/drift_engine.py` computed standard error as $SE = \sqrt{P_{\text{pool}}(1 - P_{\text{pool}})(2/N)}$. When $P_{\text{base}} = 0.0$ and $P_{\text{quant}} = 0.0$, $P_{\text{pool}} = 0.0$, causing division-by-zero crashes during $Z = (P_{\text{quant}} - P_{\text{base}}) / SE$.
* **Human Intervention:** Code review identified this issue prior to full matrix evaluation.
* **Resulting Insight:** Added epsilon floor protection (`se = 1e-6`) in `src/drift_engine.py`, enabling deterministic execution across all $N=100$ failure runs.

---

## 3. Human Reasoning Contributions

The validity of this research depended on human judgment in four key areas:

1. **Experimental Control Protocol:** Locking decoding parameters to greedy sampling ($T=0.0$, `seed=42`) and establishing an unquantized $\text{FP16}$ baseline ($P_{\text{base}} = 100.0\%$) was a human architectural choice to isolate weight precision as the sole independent variable.
2. **Dual-Pass Verification Logic:** Designing $P_{\text{raw}}$ vs. $P_{\text{scrubbed}}$ was human-driven, proving that regex markdown stripping recovered only $7.0\%$ of $0.5\text{B-INT8}$ outputs and $0.0\%$ of $1.5\text{B-INT8}$ outputs.
3. **Failure Taxonomy Definition:** Grouping errors into `SYNTAX_ERROR`, `SCHEMA_FIELD_MISMATCH`, and `SCHEMA_TYPE_MISMATCH` required manual inspection of raw token outputs to discover attention routing collapse and system prompt role erosion.
4. **Contextualizing Unconstrained Generation:** Distinguishing intrinsic weight compliance (unconstrained generation) from runtime enforcement (grammar-constrained logit processors like Outlines) was a human conceptual distinction.

---

## 4. Reflective Account on "Vibe Coding" Hazards

* **Illusion of Completeness:** AI systems default to producing authoritative summaries and paper drafts before all diagnostic evaluation steps have been run.
* **Over-Claiming Effect Sizes:** AI outputs tend to frame dramatic negative numbers (e.g., $-100.0\%$ drift) as universal laws rather than contextualizing them within specific algorithmic choices.
* **Silent Technical Debt:** Automatically generated code may pass simple tests while containing divide-by-zero or floating-point precision edge cases under extreme conditions ($0\%$ or $100\%$ pass rates).

---

## 5. Attestation of Integrity

All reported evaluation metrics ($P_{\text{raw}}$, $P_{\text{scrubbed}}$, $P_{\text{semantic}}$, $D_{\text{formatting}}$, $Z$-scores, $p$-values) were calculated from deterministic execution logs generated on local CPU hardware. No metrics, execution logs, or citations were fabricated.

**Author Signature:** Vedartha Sai Y (IIITA)  
**Date:** October 30, 2026