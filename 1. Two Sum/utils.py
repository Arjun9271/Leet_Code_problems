from fastapi import FastAPI, UploadFile, File
from typing import Dict
import shutil, os, requests, time

from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter


# ------------------------------
# 1. Custom LLM Class
# ------------------------------
class CustomLLM:
    def _call(self, context: str, question: str, instruction: str) -> str:
        prompt = f"""
You are a legal metadata extraction assistant.
Analyze the context carefully and answer the question.

Context:
{context}

Question: {question}
Instruction: {instruction}

Respond strictly according to the instruction.
Do not add any extra words or explanations.
"""
        payload = {
            "role": "Legal Metadata Extractor",
            "content": prompt,
            "question": question,
            "max_token": 512,
            "temperature": 0.1
        }

        for attempt in range(3):
            try:
                res = requests.post("http://10.20.246.200:8022/llama/inference", params=payload, timeout=60)
                res.raise_for_status()
                return res.json().get("result", "").strip()
            except Exception as e:
                print(f"[retry {attempt+1}] LLaMA call failed: {e}")
                time.sleep(2)
        raise Exception("Failed to get response after 3 attempts")

    def __call__(self, context, question, instruction):
        return self._call(context, question, instruction)

llm = CustomLLM()


# ------------------------------
# 2. FIELD_QUERIES with Query + Instruction
# ------------------------------
FIELD_QUERIES: Dict = {
    "title": {
        "query": "title or name of the agreement",
        "instruction": "Return only the title as a short phrase without extra words."
    },
    "effective_date": {
        "query": "effective date of the agreement",
        "instruction": "Return only the effective date in YYYY-MM-DD format."
    },
    "expiry_date": {
        "query": "expiry or termination date of the agreement",
        "instruction": "Return only the expiry date in YYYY-MM-DD format."
    },
    "contract_type": {
        "query": "type of the agreement",
        "instruction": "Return only one or two words (e.g., Service Agreement)."
    },
    "party_name": {
        "query": "primary party name in the agreement",
        "instruction": "Return only the main party name without any extra text."
    },
    "parties": {
        "query": "all parties involved in the agreement",
        "instruction": "Return names separated by commas only."
    },
    "key_terms_and_conditions": {
        "financial": {
            "payment_terms": {
                "query": "payment terms of the agreement",
                "instruction": "Return only the short term (e.g., 'Net 30 days')."
            }
        },
        "legal": {
            "liability_cap": {
                "query": "liability cap of the agreement",
                "instruction": "Return only the numeric value or short phrase."
            }
        },
        "performance": {
            "service_level_agreement": {
                "query": "service level agreement terms",
                "instruction": "Return in one concise sentence."
            }
        }
    }
}


# ------------------------------
# 3. VectorStore & RAG Logic
# ------------------------------
def build_vectorstore(pdf_path):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(pages)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.from_documents(chunks, embeddings)

def extract_fields(schema: Dict, vectordb, output: Dict):
    for key, value in schema.items():
        if isinstance(value, dict) and "query" in value and "instruction" in value:
            # Leaf node
            docs = vectordb.similarity_search(value["query"], k=3)
            context = "\n\n".join([d.page_content for d in docs])
            try:
                answer = llm(context, value["query"], value["instruction"])
                output[key] = answer
            except Exception:
                output[key] = ""
        else:
            # Nested dictionary
            output[key] = {}
            extract_fields(value, vectordb, output[key])

def extract_metadata(pdf_path):
    vectordb = build_vectorstore(pdf_path)
    extracted_data = {}
    extract_fields(FIELD_QUERIES, vectordb, extracted_data)
    return extracted_data


# ------------------------------
# 4. FastAPI App
# ------------------------------
app = FastAPI()
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

@app.post("/extract_fields/")
async def extract_fields_api(file: UploadFile = File(...)):
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    data = extract_metadata(file_path)
    return {"status": "success", "data": data}
