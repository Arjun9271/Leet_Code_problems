from langchain.llms.base import LLM
import requests
import time

class CustomLLM(LLM):
    def _call(self, content: str, **kwargs):
        role_prompt = """
        You are a legal assistant. Analyze the following changes between two versions of a legal document.
        Summarize the key differences clearly and concisely in 150 words. Focus on legal, factual, and structural changes.
        Do not include bullet points or markdown formatting. If there are no changes, return "No changes found".
        """
        payload = {
            'role': role_prompt,
            'content': content,
            'question': "Summarize the provided document",
            'max_token': 1024,
            'temperature': 0.3
        }

        for attempt in range(3):
            try:
                res = requests.post('http://10.20.246.200:8022/llama/inference', params=payload, timeout=60)
                res.raise_for_status()
                return res.json().get('result', '').strip()
            except Exception as e:
                print(f'[retry {attempt+1}] LLAMA call failed: {e}')
                time.sleep(2)

        return "[LLAMA ERROR] Failed after 3 retries"

    @property
    def _llm_type(self):
        return "custom-llama"

###################################
#utils.py
###########################
from langchain_community.document_loaders import PyPDFLoader
from difflib import SequenceMatcher

def extract_text_from_pdf(pdf_path: str) -> str:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    text = "\n".join([doc.page_content for doc in pages if doc.page_content and len(doc.page_content.strip()) > 50])
    return text.strip()

def get_diff_context(text1: str, text2: str) -> str:
    matcher = SequenceMatcher(None, text1.splitlines(), text2.splitlines())
    diffs = []
    for opcode, i1, i2, j1, j2 in matcher.get_opcodes():
        if opcode in ("insert", "replace", "delete"):
            diffs.append(" ".join(text2.splitlines()[j1:j2]))
    return "\n".join(diffs)
def get_diff_context(text1: str, text2: str) -> str:
    dmp = diff_match_patch()
    diffs = dmp.diff_main(text1, text2)
    dmp.diff_cleanupSemantic(diffs)

    removed = []
    added = []
    modified = []

    i = 0
    while i < len(diffs):
        if i < len(diffs) - 1 and diffs[i][0] == -1 and diffs[i+1][0] == 1:
            modified.append(f"'{diffs[i][1]}' → '{diffs[i+1][1]}'")
            i += 2
        else:
            if diffs[i][0] == -1:
                removed.append(diffs[i][1])
            elif diffs[i][0] == 1:
                added.append(diffs[i][1])
            i += 1

    diff_text = ""
    if modified:
        diff_text += "Modified:\n" + "\n".join(modified) + "\n\n"
    if removed:
        diff_text += "Removed:\n" + "\n".join(removed) + "\n\n"
    if added:
        diff_text += "Added:\n" + "\n".join(added)

    return diff_text.strip()
##########################
#fastapi
###########################
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import List
import shutil
import os

from sm_chain import CustomLLM
from utils import extract_text_from_pdf, get_diff_context

app = FastAPI()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

version_pairs = []
version_summaries = []

def save_file(file: UploadFile) -> str:
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return file_path

@app.post("/process/")
async def process(mode: str = Form(...), files: List[UploadFile] = File(...)):
    if mode == "summarize":
        if len(files) != 1:
            return JSONResponse(status_code=400, content={"error": "Upload exactly 1 file for summarize mode."})
        file_path = save_file(files[0])
        text = extract_text_from_pdf(file_path)
        summary = CustomLLM()(text)
        return {"summary": summary}

    elif mode == "difference":
        if len(files) != 2:
            return JSONResponse(status_code=400, content={"error": "Upload exactly 2 files for difference mode."})
        
        file1_path = save_file(files[0])
        file2_path = save_file(files[1])
        name1, name2 = files[0].filename, files[1].filename

        text1 = extract_text_from_pdf(file1_path)
        text2 = extract_text_from_pdf(file2_path)
        diff_text = get_diff_context(text1, text2)
        summary = CustomLLM()(diff_text)

        version_pairs.append((name1, name2))
        version_summaries.append(summary)

        return {
            "from_version": name1,
            "to_version": name2,
            "diff_summary": summary
        }

    else:
        return JSONResponse(status_code=400, content={"error": "Invalid mode. Choose 'summarize' or 'difference'."})

@app.get("/final-summary/")
async def get_final_summary():
    if not version_summaries:
        return {"summary": "No version comparisons uploaded yet."}
    
    versioned = [
        f"Changes from {v1} to {v2}:\n{summary}"
        for (v1, v2), summary in zip(version_pairs, version_summaries)
    ]
    combined_text = "\n\n".join(versioned)
    final_summary = CustomLLM()(combined_text)

    return {
        "pairwise_summaries": versioned,
        "final_summary": final_summary
    }

@app.post("/reset/")
async def reset_summary():
    version_pairs.clear()
    version_summaries.clear()
    return {"status": "Reset complete"}


