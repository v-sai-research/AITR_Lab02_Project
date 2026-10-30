import re
import json
import jsonschema
from jsonschema import validate, ValidationError as JsonSchemaValidationError
from typing import Dict, Any, Tuple

class DualPassSchemaVerifier:
    def __init__(self):
        pass

    @staticmethod
    def extract_markdown_json(text: str) -> Tuple[str, bool]:
        """
        Extracts content from ```json ... ``` or ``` ... ``` code blocks if present.
        Returns (extracted_text, had_markdown_fence).
        """
        pattern = r"```(?:json)?\s*([\s\S]*?)\s*```"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip(), True
        return text.strip(), False

    def verify_completion(
        self, raw_completion: str, json_schema: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Executes dual-pass parsing and returns verification status dictionary.
        """
        result = {
            "strict_raw_pass": False,
            "scrubbed_pass": False,
            "pass_mode": "FAIL",
            "error_category": None,
            "parsed_output": None,
            "error_message": None
        }

        # --- PASS 1: Strict Direct Parsing ---
        cleaned_raw = raw_completion.strip()
        try:
            parsed_raw = json.loads(cleaned_raw)
            # Validate against JSON schema
            validate(instance=parsed_raw, schema=json_schema)
            result["strict_raw_pass"] = True
            result["scrubbed_pass"] = True
            result["pass_mode"] = "DIRECT_PASS"
            result["parsed_output"] = parsed_raw
            return result
        except json.JSONDecodeError as e:
            raw_json_error = e
        except JsonSchemaValidationError as e:
            # Valid JSON, but schema failure
            result["error_category"] = "SCHEMA_TYPE_MISMATCH" if "type" in str(e) else "MISSING_REQUIRED_KEY"
            result["error_message"] = str(e.message)
            return result

        # --- PASS 2: Scrubbed Markdown Parsing ---
        extracted_text, had_fence = self.extract_markdown_json(raw_completion)
        if had_fence:
            try:
                parsed_scrubbed = json.loads(extracted_text)
                validate(instance=parsed_scrubbed, schema=json_schema)
                result["strict_raw_pass"] = False
                result["scrubbed_pass"] = True
                result["pass_mode"] = "SCRUBBED_PASS"
                result["error_category"] = "MARKDOWN_FENCE_VIOLATION"
                result["parsed_output"] = parsed_scrubbed
                return result
            except json.JSONDecodeError as e:
                result["error_category"] = "SYNTAX_ERROR"
                result["error_message"] = f"Scrubbed JSON error: {e}"
                return result
            except JsonSchemaValidationError as e:
                result["error_category"] = "SCHEMA_FIELD_MISMATCH"
                result["error_message"] = str(e.message)
                return result

        # If no markdown fence was present and raw JSON failed
        result["error_category"] = "SYNTAX_ERROR"
        result["error_message"] = f"Direct JSON error: {raw_json_error}"
        return result