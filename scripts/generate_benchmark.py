import os
import json

def generate_json_schema_micro_suite():
    os.makedirs("data", exist_ok=True)
    benchmark_path = os.path.join("data", "json_schema_eval_100.jsonl")
    
    prompts = []
    
    # Tier 1: Primitive Flat Schemas (35 items)
    for i in range(1, 36):
        prompts.append({
            "id": f"tier1_flat_{i:03d}",
            "schema_tier": "Tier 1: Primitive Flat",
            "context": f"User ID u_{1000+i} registered under email user{i}@research.org with active status {True if i % 2 == 0 else False} and score {i * 2.5}.",
            "pydantic_schema": {
                "title": "UserProfile",
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "email": {"type": "string"},
                    "is_active": {"type": "boolean"},
                    "score": {"type": "number"}
                },
                "required": ["user_id", "email", "is_active", "score"]
            },
            "ground_truth": {
                "user_id": f"u_{1000+i}",
                "email": f"user{i}@research.org",
                "is_active": i % 2 == 0,
                "score": i * 2.5
            }
        })
        
    # Tier 2: Nested Object Hierarchies (35 items)
    for i in range(1, 36):
        prompts.append({
            "id": f"tier2_nested_{i:03d}",
            "schema_tier": "Tier 2: Nested Object",
            "context": f"Corporate entity Org_{i} located in Department Dept_{i%5} managed by Director_Key_{i} with budget {50000 + i*1000}.",
            "pydantic_schema": {
                "title": "CorporateOrg",
                "type": "object",
                "properties": {
                    "org_id": {"type": "string"},
                    "department": {
                        "type": "object",
                        "properties": {
                            "dept_name": {"type": "string"},
                            "manager_id": {"type": "string"},
                            "budget": {"type": "number"}
                        },
                        "required": ["dept_name", "manager_id", "budget"]
                    }
                },
                "required": ["org_id", "department"]
            },
            "ground_truth": {
                "org_id": f"Org_{i}",
                "department": {
                    "dept_name": f"Dept_{i%5}",
                    "manager_id": f"Director_Key_{i}",
                    "budget": 50000 + i * 1000
                }
            }
        })

    # Tier 3: Typed Array Collections (30 items)
    for i in range(1, 31):
        prompts.append({
            "id": f"tier3_array_{i:03d}",
            "schema_tier": "Tier 3: Typed Array",
            "context": f"Patient P_{i:03d} holds prescriptions: Med_A (dose {i*10}mg) and Med_B (dose {i*5}mg).",
            "pydantic_schema": {
                "title": "MedicalPrescription",
                "type": "object",
                "properties": {
                    "patient_id": {"type": "string"},
                    "medications": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "med_name": {"type": "string"},
                                "dose_mg": {"type": "number"}
                            },
                            "required": ["med_name", "dose_mg"]
                        }
                    }
                },
                "required": ["patient_id", "medications"]
            },
            "ground_truth": {
                "patient_id": f"P_{i:03d}",
                "medications": [
                    {"med_name": "Med_A", "dose_mg": float(i * 10)},
                    {"med_name": "Med_B", "dose_mg": float(i * 5)}
                ]
            }
        })

    with open(benchmark_path, "w", encoding="utf-8") as f:
        for item in prompts:
            f.write(json.dumps(item) + "\n")
            
    print(f"[+] Successfully generated {len(prompts)} stratified benchmark prompts -> {benchmark_path}")

if __name__ == "__main__":
    generate_json_schema_micro_suite()