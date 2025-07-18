from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from difflib import SequenceMatcher
def extract_text_from_pdf(pdf_path: str) -> str:
    loader = PyPDFLoader(pdf_path)
    pages = loader.load()
    text = "\n".join([doc.page_content for doc in pages if doc.page_content and len(doc.page_content.strip()) > 50])
    return text.strip()


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

