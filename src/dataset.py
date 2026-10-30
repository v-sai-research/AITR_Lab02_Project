import os
import json
from typing import List, Dict, Any

class BenchmarkDatasetLoader:
    REQUIRED_KEYS = {"id", "schema_tier", "context", "pydantic_schema", "ground_truth"}
    EXPECTED_TIERS = {
        "Tier 1: Primitive Flat": 35,
        "Tier 2: Nested Object": 35,
        "Tier 3: Typed Array": 30
    }

    def __init__(self, filepath: str = "data/json_schema_eval_100.jsonl"):
        self.filepath = filepath
        self.data: List[Dict[str, Any]] = []

    def load_and_validate(self) -> List[Dict[str, Any]]:
        if not os.path.exists(self.filepath):
            raise FileNotFoundError(f"Dataset file not found at: {self.filepath}")

        loaded_items = []
        tier_counts = {tier: 0 for tier in self.EXPECTED_TIERS}

        with open(self.filepath, "r", encoding="utf-8") as f:
            for line_idx, line in enumerate(f, 1):
                line_str = line.strip()
                if not line_str:
                    continue
                
                try:
                    record = json.loads(line_str)
                except json.JSONDecodeError as e:
                    raise ValueError(f"Line {line_idx}: Invalid JSON string -> {e}")

                # Validate required schema keys
                missing_keys = self.REQUIRED_KEYS - record.keys()
                if missing_keys:
                    raise KeyError(f"Line {line_idx} (ID: {record.get('id', 'UNKNOWN')}): Missing required keys {missing_keys}")

                tier = record["schema_tier"]
                if tier in tier_counts:
                    tier_counts[tier] += 1
                else:
                    raise ValueError(f"Line {line_idx}: Unknown tier category '{tier}'")

                loaded_items.append(record)

        # Validate total count
        if len(loaded_items) != 100:
            raise ValueError(f"Dataset size mismatch: Expected 100 records, got {len(loaded_items)}")

        # Validate tier stratification counts
        for tier_name, expected_count in self.EXPECTED_TIERS.items():
            if tier_counts[tier_name] != expected_count:
                raise ValueError(
                    f"Stratification error in '{tier_name}': Expected {expected_count}, got {tier_counts[tier_name]}"
                )

        self.data = loaded_items
        return self.data

    def get_partition_by_tier(self, tier_name: str) -> List[Dict[str, Any]]:
        return [item for item in self.data if item["schema_tier"] == tier_name]