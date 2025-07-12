

import fitz
import torch
import json
from datetime import datetime
from transformers import AutoTokenizer, AutoModelForCausalLM

# ✅ Correct model and tokenizer loading for TEXT mode
model_id = "numind/NuExtract-2.0-8B"
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    trust_remote_code=True,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)
model.eval()

# ✅ Schema template using NuMind style
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

# ✅ Extract plain text from PDF
def extract_text(pdf_path):
    doc = fitz.open(pdf_path)
    return "\n".join(page.get_text() for page in doc)

# ✅ Token-level chunking
def chunk_text(text, max_tokens=1500):
    tokens = tokenizer.encode(text)
    return [tokenizer.decode(tokens[i:i+max_tokens]) for i in range(0, len(tokens), max_tokens)]

# ✅ Run NuExtract model on each chunk
def extract_chunk(text_chunk):
    prompt = {
        "template": template,
        "input": text_chunk
    }
    inputs = tokenizer(json.dumps(prompt), return_tensors="pt", truncation=True, max_length=4096).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512, temperature=0)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return json.loads(decoded)

# ✅ Merge outputs from all chunks
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

# ✅ Auto-calculate duration from dates
def calculate_duration(start, end):
    try:
        d1 = datetime.fromisoformat(start)
        d2 = datetime.fromisoformat(end)
        months = round((d2 - d1).days / 30.44)
        return f"{months} months"
    except:
        return None

# ✅ Final pipeline
def extract_contract(pdf_path):
    text = extract_text(pdf_path)
    chunks = chunk_text(text)
    outputs = [extract_chunk(c) for c in chunks]
    merged = merge_results(outputs)
    if not merged.get("duration") and merged.get("effective_date") and merged.get("expiry_date"):
        dur = calculate_duration(merged["effective_date"], merged["expiry_date"])
        if dur: merged["duration"] = dur
    return merged

# 🔧 Example run
if __name__ == "__main__":
    result = extract_contract("your_contract.pdf")  # Replace with your PDF
    print(json.dumps(result, indent=2))
