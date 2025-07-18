##################main.py##############################################
from fastapi import FastAPI, UploadFile, File, Form, APIRouter
from fastapi.responses import JSONResponse
from cleanup import cleanup_old_files
from typing import List,Literal
import uvicorn
import shutil
import json
import os

from sm_chain import CustomLLM
from utils import (
    extract_text_from_pdf,
    get_diff_context,
    load_and_chunk_pdf,
    format_diff_for_prompt,
    extract_diff_pairs_by_type,
    compare_pdfs_flat_items 
)


app = FastAPI()
router = APIRouter()

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

version_pairs = []
version_summaries = []

def save_file(file: UploadFile) -> str:
    file_path = os.path.join(UPLOAD_DIR, file.filename)
    with open(file_path, "wb") as f:
        shutil.copyfileobj(file.file, f)
    return file_path


@router.post("/extract_fields/")
async def extract_fields(files: List[UploadFile] = File(...)):
    cleanup_old_files()
    if len(files) != 1:
        return JSONResponse(status_code=400, content={"error": "Upload exactly 1 file to extract fields."})
    
    file_path = save_file(files[0])
    chunks = load_and_chunk_pdf(file_path)
    
    try:
        text = extract_text_from_pdf(file_path)
        response = CustomLLM()(text, mode='extract')
        parsed = json.loads(response)
        return {"extracted_fields": parsed}
    
    except Exception as e:
        fields = [CustomLLM()(doc.page_content, mode='extract') for doc in chunks]
        return {"chunked_field_responses": fields}






@router.post("/process/")
async def process(mode: Literal["summarize", "difference"] = Form(...), files: List[UploadFile] = File(...)):
    cleanup_old_files()
    if mode == "summarize":
        if len(files) != 1:
            return JSONResponse(status_code=400, content={"error": "Upload exactly 1 file for summarize mode."})
        file_path = save_file(files[0])
        chunks = load_and_chunk_pdf(file_path)
        try:
            text= extract_text_from_pdf(file_path)
            summary = CustomLLM()(text,mode = 'summarize')
            return {"summary": summary}

        except Exception as e:
            summaries = [CustomLLM()(doc.page_content,mode = 'summarize') for doc in chunks]
            combined = '\n\n'.join(summaries)
            summary = CustomLLM()(combined,mode="summarize")
            return {"summary": summary}
    
    
    
    elif mode == "difference":
            if len(files) != 2:
                return JSONResponse(status_code=400, content={"error": "Upload exactly 2 files for difference mode."})

            file1_path = save_file(files[0])
            file2_path = save_file(files[1])
            name1, name2 = files[0].filename, files[1].filename
            # Step 1: flat diff output with page numbers
            flat_diffs = compare_pdfs_flat_items(file1_path, file2_path)

            # Step 2: reuse merged text for summary formatting
            text1 = extract_text_from_pdf(file1_path)
            text2 = extract_text_from_pdf(file2_path)
            summary_input = format_diff_for_prompt(get_diff_context(text1, text2))
            summary = CustomLLM()(summary_input, mode="difference")

            version_pairs.append((name1, name2))
            version_summaries.append(summary)


            # Return response
            return {
                "from_version": name1,
                "to_version": name2,
                "diff_summary": summary,
                "diff_items": flat_diffs  
            }


    else:
            return JSONResponse(status_code=400, content={"error": "Invalid mode. Choose 'summarize' or 'difference'."})

@router.get("/final_summary/")
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

@router.post("/reset/")
async def reset_summary():
    cleanup_old_files()
    version_pairs.clear()
    version_summaries.clear()
    return {"status": "Reset complete"}




app.include_router(router, prefix="/doc", tags=["Document_Summarization"])


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8356)




################################utils.py##########################################

from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from difflib import SequenceMatcher
def extract_text_from_pdf(pdf_path: str) -> str:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    text = "\n".join([doc.page_content for doc in pages if doc.page_content and len(doc.page_content.strip()) > 50])
    return text.strip()



from langchain_community.document_loaders import PyPDFLoader

def extract_pages(pdf_path: str, min_len: int = 50):
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    extracted = []
    for doc in docs:
        text = doc.page_content.strip() if doc.page_content else ""
        if len(text) > min_len:
            extracted.append({
                "text": text,
                "page": (doc.metadata.get("page", 0) or 0) + 1
            })
    return extracted


def compare_pdfs_flat_items(pdf1_path: str, pdf2_path: str) -> list:
    pages1 = extract_pages(pdf1_path)
    pages2 = extract_pages(pdf2_path)

    max_len = max(len(pages1), len(pages2))
    flat_result = []

    for i in range(max_len):
        text1 = pages1[i]["text"] if i < len(pages1) else ""
        text2 = pages2[i]["text"] if i < len(pages2) else ""
        page = pages2[i].get("page") if i < len(pages2) else pages1[i].get("page", i + 1)

        diff = get_diff_context(text1, text2)

        for tag, items in diff.items():
            for item in items:
                entry = { "type": tag, "page": page }
                if tag == "MODIFIED":
                    entry.update({
                        "from": item["old_text"],
                        "to": item["new_text"]
                    })
                else:
                    entry["text"] = item["text"]
                flat_result.append(entry)

    return flat_result






def load_and_chunk_pdf(pdf_path,chunk_size = 12000,chunk_overlap = 2000):
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        separators = ['\n\n','\n','.',' ','']
    )

    chunks = splitter.split_documents(pages)
    return chunks





from difflib import SequenceMatcher

def get_diff_context(text1: str, text2: str) -> dict:
    words1 = text1.split()
    words2 = text2.split()
    lower_words1 = [w.lower() for w in words1]
    lower_words2 = [w.lower() for w in words2]

    matcher = SequenceMatcher(None, lower_words1, lower_words2)

    categorized_diff = {
        "ADDED": [],
        "REMOVED": [],
        "MODIFIED": [],
        "UNCHANGED": []
    }

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            categorized_diff["UNCHANGED"].append({
                "text": ' '.join(words1[i1:i2])
            })
        elif tag == "insert":
            categorized_diff["ADDED"].append({
                "text": ' '.join(words2[j1:j2])
            })
        elif tag == "delete":
            categorized_diff["REMOVED"].append({
                "text": ' '.join(words1[i1:i2])
            })
        elif tag == "replace":
            categorized_diff["MODIFIED"].append({
                "old_text": ' '.join(words1[i1:i2]),
                "new_text": ' '.join(words2[j1:j2])
            })

    return categorized_diff







def format_diff_for_prompt(diff_dict: dict) -> str:
    sections = []
    for key, diffs in diff_dict.items():
        if not diffs:
            continue
        section_lines = [f"{key}:"]
        for item in diffs:
            if key == "MODIFIED":
                section_lines.append(f"- From: {item['old_text']}\n  To:   {item['new_text']}")
            elif key in ["ADDED", "REMOVED", "UNCHANGED"]:
                section_lines.append(f"- {item['text']}")
            else:
                section_lines.append(f"- {item}")
        sections.append('\n'.join(section_lines))
    return '\n\n'.join(sections)



def extract_diff_pairs_by_type(diff_dict: dict) -> dict:
    categorized_pairs = {
        "ADDED": [],
        "REMOVED": [],
        "MODIFIED": [],
        "UNCHANGED": []
    }

    for key, items in diff_dict.items():
        for item in items:
            if key == "MODIFIED":
                categorized_pairs["MODIFIED"].append({
                    "from": item["old_text"],
                    "to": item["new_text"]
                })
            else:
                categorized_pairs[key].append({
                    "text": item["text"]
                })

    return categorized_pairs


