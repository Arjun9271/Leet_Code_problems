import fitz
import json
from datetime import datetime
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForVision2Seq

# Load model and processor
model_id = "numind/NuExtract-2.0-2B"  # or -8B if your GPU supports it
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForVision2Seq.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",  # Optional, improves performance
    device_map="auto",
    trust_remote_code=True
)
model.eval()

# Contract field schema
template = {
    "title": "string",
    "effective_date": "date-time",
    "expiry_date": "date-time",
    "party_name": "string",
    "contract_type": "string",
    "parties_involved": ["string"],
    "duration": "string",
    "key_terms": {
        "financial": {"payment_terms": "string"},
        "legal": {"liability_cap": "string"},
        "performance": {"service_level_agreement": "string"}
    }
}

# Extract plain text from PDF
def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)

# Token-aware chunking
def chunk_text(text, tokenizer, max_tokens=1500):
    tokens = tokenizer.encode(text)
    return [tokenizer.decode(tokens[i:i+max_tokens]) for i in range(0, len(tokens), max_tokens)]

# Format input using chat_template
def format_prompt(text_chunk, processor, template):
    messages = [{"role": "user", "content": text_chunk}]
    return processor.tokenizer.apply_chat_template(
        messages,
        template=template,
        tokenize=False,
        add_generation_prompt=True
    )

# Inference per chunk
def extract_chunk(text_chunk):
    prompt = format_prompt(text_chunk, processor, template)
    inputs = processor(text=[prompt], return_tensors="pt", padding=True).to(model.device)

    output_ids = model.generate(
        **inputs,
        max_new_tokens=1024,
        do_sample=False,
        num_beams=1
    )

    response = processor.batch_decode(output_ids, skip_special_tokens=True)[0]
    try:
        return json.loads(response)
    except:
        print("⚠️ Warning: Failed to parse JSON from response:")
        print(response)
        return {}

# Merge chunk outputs
def merge_results(results):
    final = {
        **template,
        "parties_involved": [],
        "key_terms": template["key_terms"]
    }
    for r in results:
        for k, v in r.items():
            if v:
                if isinstance(v, list):
                    final[k] = list(set(final.get(k, []) + v))
                elif isinstance(v, dict):
                    final[k].update({**final[k], **v})
                else:
                    final[k] = final.get(k) or v
    return final

# Duration calculator
def calculate_duration(start, end):
    try:
        d1 = datetime.fromisoformat(start)
        d2 = datetime.fromisoformat(end)
        months = round((d2 - d1).days / 30.44)
        return f"{months} months"
    except:
        return None

# Final pipeline
def extract_contract(pdf_path):
    text = extract_text(pdf_path)
    chunks = chunk_text(text, processor.tokenizer)
    extracted = [extract_chunk(c) for c in chunks]
    merged = merge_results(extracted)

    if not merged.get("duration") and merged.get("effective_date") and merged.get("expiry_date"):
        duration = calculate_duration(merged["effective_date"], merged["expiry_date"])
        if duration:
            merged["duration"] = duration

    return merged

# Run it
if __name__ == "__main__":
    result = extract_contract("your_contract.pdf")
    print(json.dumps(result, indent=2))
