"""Hugging Face inference providers used by the campaign graph."""
import json
import os
import re
import time
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

TEXT_MODEL = os.getenv("HF_TEXT_MODEL", "Qwen/Qwen3-8B")
VISION_MODEL = os.getenv("HF_VISION_MODEL", "Qwen/Qwen3-VL-8B-Instruct")
IMAGE_MODEL = os.getenv("HF_IMAGE_MODEL", "black-forest-labs/FLUX.1-Krea-dev")
HF_TOKEN = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_HUB_TOKEN")
MODEL_DTYPE = os.getenv("HF_DTYPE", "bfloat16")
QUANTIZE_4BIT = os.getenv("HF_QUANTIZE_4BIT", "true").lower() in {"1", "true", "yes"}
DEVICE = os.getenv("HF_DEVICE", "cuda")
_text = _vision = _image = None


def _dtype():
    import torch
    return torch.float16 if MODEL_DTYPE in {"float16", "fp16"} else torch.bfloat16


def _quantization_config():
    if not QUANTIZE_4BIT:
        return None
    from transformers import BitsAndBytesConfig
    return BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=_dtype(), bnb_4bit_quant_type="nf4")


def _load_text():
    global _text
    if _text is not None:
        return _text
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(TEXT_MODEL, token=HF_TOKEN)
    kwargs = {"torch_dtype": _dtype(), "token": HF_TOKEN, "device_map": "auto"}
    quant = _quantization_config()
    if quant:
        kwargs["quantization_config"] = quant
    model = AutoModelForCausalLM.from_pretrained(TEXT_MODEL, **kwargs)
    _text = tokenizer, model
    return _text


def _text_generate(prompt: str) -> str:
    tokenizer, model = _load_text()
    messages = [{"role": "user", "content": prompt}]
    inputs = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                            enable_thinking=False, return_tensors="pt").to(model.device)
    output = model.generate(inputs, max_new_tokens=int(os.getenv("HF_MAX_NEW_TOKENS", "4096")),
                            temperature=0.4, do_sample=True, top_p=0.9)
    return tokenizer.decode(output[0][inputs.shape[-1]:], skip_special_tokens=True).strip()


def _call_with_retry(fn, max_attempts=3, base_delay=2):
    last = None
    for attempt in range(max_attempts):
        try:
            return fn()
        except Exception as exc:
            last = exc
            if attempt < max_attempts - 1:
                time.sleep(base_delay * (attempt + 1))
    raise last


def _repair_json(text: str) -> str:
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S).strip()
    start, end = text.find("{"), text.rfind("}")
    if start >= 0 and end > start:
        text = text[start:end + 1]
    return re.sub(r",\s*([}\]])", r"\1", text)


def ask_json(system_prompt: str, user_prompt: str, model: str = None) -> dict:
    """Ask Qwen3-8B for a structured JSON response."""
    prompt = (system_prompt + "\n\nIMPORTANT: Respond ONLY with one valid JSON object. "
              "No markdown or explanation.\n\n" + user_prompt)
    for _ in range(3):
        try:
            raw = _call_with_retry(lambda: _text_generate(prompt))
            text = _repair_json(raw)
            try:
                return json.loads(text)
            except json.JSONDecodeError:
                for suffix in ("}", "]}", "}}"):
                    try:
                        return json.loads(text + suffix)
                    except json.JSONDecodeError:
                        pass
        except Exception:
            pass
    return {}


def generate_image(prompt: str, output_path: str) -> str:
    """Generate one image with FLUX.1-Krea-dev."""
    global _image
    if _image is None:
        from diffusers import DiffusionPipeline
        _image = DiffusionPipeline.from_pretrained(IMAGE_MODEL, torch_dtype=_dtype(), token=HF_TOKEN)
        if DEVICE == "cuda":
            _image.enable_model_cpu_offload()
        else:
            _image.to(DEVICE)
    image = _image(prompt=prompt, height=int(os.getenv("HF_IMAGE_HEIGHT", "1024")),
                   width=int(os.getenv("HF_IMAGE_WIDTH", "1024")),
                   guidance_scale=float(os.getenv("HF_IMAGE_GUIDANCE", "4.5")),
                   num_inference_steps=int(os.getenv("HF_IMAGE_STEPS", "28"))).images[0]
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    image.save(output_path)
    return output_path


def evaluate_image(image_path: str, requirements: dict) -> dict:
    """Review an image with Qwen3-VL and return validated JSON."""
    global _vision
    if _vision is None:
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
        processor = AutoProcessor.from_pretrained(VISION_MODEL, token=HF_TOKEN)
        kwargs = {"torch_dtype": _dtype(), "device_map": "auto", "token": HF_TOKEN}
        quant = _quantization_config()
        if quant:
            kwargs["quantization_config"] = quant
        _vision = processor, Qwen3VLForConditionalGeneration.from_pretrained(VISION_MODEL, **kwargs)
    processor, model = _vision
    prompt = ("Inspect the attached image against these campaign requirements. Return ONLY JSON with "
              "score (integer 0-100), issues (array of strings), critical_issues (array of strings), "
              "and suggestions (array of strings). Check subject recognition, visual goals, composition, "
              "anatomy, misleading claims, legibility, safe areas, and brand alignment. Do not reject "
              "caption-only goals. Requirements: " + json.dumps(requirements, ensure_ascii=False))
    messages = [{"role": "user", "content": [{"type": "image", "image": image_path},
                                                  {"type": "text", "text": prompt}]}]
    text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[image_path], return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=700, do_sample=False)
    raw = processor.batch_decode(output[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)[0]
    result = json.loads(_repair_json(raw))
    if not isinstance(result, dict) or type(result.get("score")) is not int or not 0 <= result["score"] <= 100:
        raise ValueError("Vision review was incomplete; campaign was not approved.")
    for key in ("issues", "critical_issues", "suggestions"):
        if not isinstance(result.get(key), list) or not all(isinstance(x, str) for x in result[key]):
            raise ValueError("Vision review returned invalid issue fields.")
    return result


def _ollama_generate(prompt: str) -> str:
    """Compatibility alias for the research module during migration."""
    return _text_generate(prompt)
