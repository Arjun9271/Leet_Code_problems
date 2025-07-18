from langchain_community.document_loaders import PyPDFLoader
from difflib import SequenceMatcher
import json


# ---------------------------
# 1. Extract text by page
# ---------------------------
def extract_pages(pdf_path: str, min_len: int = 50):
    """
    Extract pages with their page number and text.
    """
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()

    page_texts = []
    for doc in pages:
        content = (doc.page_content or "").strip()
        if content and len(content) > min_len:
            page_texts.append({
                "page": doc.metadata.get("page", None),
                "text": content
            })
    return page_texts


# ---------------------------
# 2. Diff two texts (word-based)
# ---------------------------
def get_diff_context(text1: str, text2: str) -> dict:
    words1 = text1.split()
    words2 = text2.split()
    matcher = SequenceMatcher(None, [w.lower() for w in words1], [w.lower() for w in words2])

    diff = {"ADDED": [], "REMOVED": [], "MODIFIED": [], "UNCHANGED": []}

    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            diff["UNCHANGED"].append({"text": " ".join(words1[i1:i2])})
        elif tag == "insert":
            diff["ADDED"].append({"text": " ".join(words2[j1:j2])})
        elif tag == "delete":
            diff["REMOVED"].append({"text": " ".join(words1[i1:i2])})
        elif tag == "replace":
            diff["MODIFIED"].append({
                "old_text": " ".join(words1[i1:i2]),
                "new_text": " ".join(words2[j1:j2])
            })
    return diff


# ---------------------------
# 3. Compare page by page
# ---------------------------
def compare_pdfs_pagewise(pdf1: str, pdf2: str):
    pages1 = extract_pages(pdf1)
    pages2 = extract_pages(pdf2)

    max_len = max(len(pages1), len(pages2))
    result = []

    for i in range(max_len):
        page_num = (pages2[i]["page"] if i < len(pages2) else pages1[i]["page"]) or (i + 1)
        text1 = pages1[i]["text"] if i < len(pages1) else ""
        text2 = pages2[i]["text"] if i < len(pages2) else ""

        page_diff = get_diff_context(text1, text2)
        result.append({
            "page": page_num,
            **page_diff
        })

    return result


# ---------------------------
# 4. Example usage
# ---------------------------
if __name__ == "__main__":
    pdf_old = "old_version.pdf"
    pdf_new = "new_version.pdf"

    diff_result = compare_pdfs_pagewise(pdf_old, pdf_new)

    print(json.dumps(diff_result, indent=2))
