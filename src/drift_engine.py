import math
import scipy.stats as stats
from typing import Dict, List, Any

class FormattingDriftEngine:
    def __init__(self, sample_size: int = 100):
        self.n = sample_size

    def compute_pass_rate(self, verification_results: List[Dict[str, Any]], pass_key: str = "scrubbed_pass") -> float:
        if not verification_results:
            return 0.0
        successes = sum(1 for r in verification_results if r.get(pass_key, False))
        return float(successes / len(verification_results))

    def compute_semantic_accuracy(self, verification_results: List[Dict[str, Any]], ground_truths: List[Dict[str, Any]]) -> float:
        if not verification_results or len(verification_results) != len(ground_truths):
            return 0.0
        
        matches = 0
        for res, gt in zip(verification_results, ground_truths):
            parsed = res.get("parsed_output")
            if parsed and parsed == gt:
                matches += 1
        return float(matches / len(verification_results))

    def calculate_drift_and_significance(
        self, p_base: float, p_quant: float, semantic_base: float = 1.0, semantic_quant: float = 1.0
    ) -> Dict[str, Any]:
        if p_base == 0:
            d_formatting = 0.0
        else:
            d_formatting = ((p_quant - p_base) / p_base) * 100.0

        x_base = round(p_base * self.n)
        x_quant = round(p_quant * self.n)
        
        p_pool = (x_base + x_quant) / (2 * self.n)
        
        if p_pool == 0 or p_pool == 1:
            se = 1e-6
        else:
            se = math.sqrt(p_pool * (1 - p_pool) * (2 / self.n))

        z_stat = (p_quant - p_base) / se
        raw_p_value = float(2 * (1 - stats.norm.cdf(abs(z_stat))))

        if semantic_base == 0:
            d_semantic = 1e-6
        else:
            d_semantic = ((semantic_quant - semantic_base) / semantic_base) * 100.0

        disparity_ratio = abs(d_formatting / d_semantic) if d_semantic != 0 else abs(d_formatting)

        return {
            "p_base": float(p_base),
            "p_quant": float(p_quant),
            "d_formatting_percent": float(round(d_formatting, 2)),
            "z_statistic": float(round(z_stat, 4)),
            "p_value": float(round(raw_p_value, 4)),
            "is_significant_p005": bool(raw_p_value < 0.05),  # Native Python bool conversion
            "disparity_ratio": float(round(disparity_ratio, 2))
        }