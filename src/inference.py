import os
import json
import torch
import random
import numpy as np
from typing import Dict, Any, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer

def set_deterministic_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

class LocalInferenceEngine:
    SYSTEM_PROMPT = (
        "You are a deterministic JSON extraction engine. "
        "You must read the input context and extract information adhering strictly to the provided JSON schema. "
        "Output ONLY valid JSON. Do not include markdown formatting, preambles, or explanations."
    )

    def __init__(self, model_id: str = "Qwen/Qwen2.5-0.5B-Instruct", device: str = "cpu", quant_bits: Optional[int] = None):
        self.model_id = model_id
        self.device = device
        self.quant_bits = quant_bits  # None (FP16/FP32), 8 (INT8 Dynamic), 4 (INT4)
        self.tokenizer = None
        self.model = None

    def load_model(self):
        """Loads model weights and applies optional dynamic quantization without breaking baseline."""
        print(f"[+] Initializing tokenizer and model: '{self.model_id}' (Quant: {self.quant_bits}-bit) on {self.device}...")
        set_deterministic_seed(42)
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, 
            trust_remote_code=True
        )
        
        # Load baseline model
        base_model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            torch_dtype=torch.float32 if self.device == "cpu" else torch.float16,
            device_map=self.device,
            trust_remote_code=True
        )

        # Apply Dynamic Quantization if requested
        if self.quant_bits == 8 and self.device == "cpu":
            print("[+] Applying Dynamic INT8 Quantization to Linear Layers...")
            self.model = torch.ao.quantization.quantize_dynamic(
                base_model, {torch.nn.Linear}, dtype=torch.qint8
            )
        else:
            self.model = base_model

        print(f"[+] Model successfully loaded and ready for inference.")

    def format_user_prompt(self, context: str, json_schema: Dict[str, Any]) -> str:
        schema_str = json.dumps(json_schema, indent=2)
        return f"Context:\n{context}\n\nRequired JSON Schema:\n{schema_str}\n\nExtracted JSON:"

    def generate_completion(
        self, context: str, json_schema: Dict[str, Any], max_new_tokens: int = 512
    ) -> str:
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model is not initialized. Call load_model() before inference.")

        set_deterministic_seed(42)
        user_prompt = self.format_user_prompt(context, json_schema)
        
        messages = [
            {"role": "system", "content": self.SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt}
        ]

        text_input = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = self.tokenizer([text_input], return_tensors="pt").to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **model_inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id
            )

        generated_ids = [
            output_ids[len(input_ids):] 
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response.strip()