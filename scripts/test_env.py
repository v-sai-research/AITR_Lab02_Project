import os
import sys
import random
import numpy as np
import torch
import psutil
from pydantic import BaseModel, Field, ValidationError

def set_deterministic_seeds(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

class EnvironmentTestSchema(BaseModel):
    test_id: str = Field(description="Unique identifier")
    status: str = Field(description="Execution status")
    ram_gb: float = Field(description="System RAM in GB")
    is_pydantic_v2: bool = Field(description="True if Pydantic 2.x")

def run_environment_test():
    print("=" * 60)
    print("      RESEARCH ENVIRONMENT DIAGNOSTIC & VERIFICATION SHEET")
    print("=" * 60)
    
    # 1. System & Memory Information
    ram_gb = round(psutil.virtual_memory().total / (1024 ** 3), 2)
    python_ver = sys.version.split()[0]
    print(f"[+] Python Version: {python_ver}")
    print(f"[+] System RAM: {ram_gb} GB")
    
    # 2. Hardware Acceleration Diagnostics
    cuda_avail = torch.cuda.is_available()
    device_name = torch.cuda.get_device_name(0) if cuda_avail else "CPU Only / No CUDA"
    print(f"[+] PyTorch Version: {torch.__version__}")
    print(f"[+] CUDA Acceleration: {cuda_avail} ({device_name})")
    
    # 3. Seed Control Freeze
    set_deterministic_seeds(42)
    rand_val = round(random.random(), 6)
    print(f"[+] Random Seed Frozen (Seed 42 -> Check Val: {rand_val})")
    assert rand_val == 0.639427, f"Seed mismatch! Expected 0.639427, got {rand_val}"
    
    # 4. Pydantic v2 Functional Check
    import pydantic
    pydantic_v2 = pydantic.__version__.startswith("2.")
    print(f"[+] Pydantic Version: {pydantic.__version__} (v2 check: {pydantic_v2})")
    assert pydantic_v2, "Pydantic 2.x is required for this experiment!"
    
    # 5. Schema Validation Engine Dummy Test
    dummy_payload = {
        "test_id": "env_001",
        "status": "PASS",
        "ram_gb": ram_gb,
        "is_pydantic_v2": pydantic_v2
    }
    
    try:
        validated = EnvironmentTestSchema(**dummy_payload)
        print("[+] Pydantic Schema Parsing Test: SUCCESS")
        print(f"    Validated Payload: {validated.model_dump()}")
    except ValidationError as e:
        print(f"[-] Pydantic Schema Parsing Test: FAILED ({e})")
        sys.exit(1)
        
    print("=" * 60)
    print("VERIFICATION RESULT: ALL ENVIRONMENT CHECKS PASSED SUCCESSFULLY.")
    print("=" * 60)

if __name__ == "__main__":
    run_environment_test()