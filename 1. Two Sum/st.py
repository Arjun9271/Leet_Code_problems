import os
import re
import shutil
import tempfile
import difflib
import pdfplumber
import pandas as pd
from typing import List, Dict

# ------------------ PDF Text Extraction ------------------ #
def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text(x_tolerance=2, y_tolerance=2) + "\n"
    return text

# ------------------ Section Extraction ------------------ #
def extract_sections_generic(text: str) -> List[Dict[str, str]]:
    lines = [ln.strip() for ln in text.splitlines()]
    lines = [ln for ln in lines if ln]  # remove empty lines

    sections = []
    current_heading = "Document Preamble"
    current_content = []

    def is_heading(line: str) -> bool:
        if len(line) < 3: 
            return False
        if line.endswith(":") and sum(c.isalpha() for c in line) >= 3:
            return True
        if line.isupper() and len(line.split()) <= 12:  # ALL CAPS short lines
            return True
        if re.match(r'^(exhibit|nature of|services|terms|payment|non[- ]|agreement|definitions)', line, re.I):
            return True
        if re.match(r'^section\s+\d+', line, re.I):
            return True
        return False

    for ln in lines:
        if is_heading(ln):
            if current_content:
                sections.append({"heading": current_heading, "content": " ".join(current_content).strip()})
                current_content = []
            current_heading = ln
        else:
            current_content.append(ln)

    if current_content:
        sections.append({"heading": current_heading, "content": " ".join(current_content).strip()})

    return sections

# ------------------ Cleaning & Normalization ------------------ #
def clean_clause_name(name: str) -> str:
    name = name.strip()
    name = re.sub(r'^[^A-Za-z0-9]+', '', name)  # strip leading junk
    name = re.sub(r'[\s]+', ' ', name)
    return name

def clause_join_key(name: str) -> str:
    key = name.lower()
    key = re.sub(r'[\W_]+', ' ', key)
    return " ".join(key.split()[:5])  # first 5 words

# ------------------ Diff Computation ------------------ #
def compare_clauses_diff(text1: str, text2: str) -> Dict[str, List]:
    words1 = text1.split()
    words2 = text2.split()
    lower_words1 = [w.lower() for w in words1]
    lower_words2 = [w.lower() for w in words2]

    sm = difflib.SequenceMatcher(None, lower_words1, lower_words2)
    categorized_diff = {"add_in": [], "omit": [], "substitute": []}

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            continue
        elif tag == 'insert':
            categorized_diff["add_in"].append(" ".join(words2[j1:j2]))
        elif tag == 'delete':
            categorized_diff["omit"].append(" ".join(words1[i1:i2]))
        elif tag == 'replace':
            categorized_diff["substitute"].append({
                "old": " ".join(words1[i1:i2]),
                "new": " ".join(words2[j1:j2])
            })

    return categorized_diff

def mark_diff(text1: str, text2: str) -> (str, str):
    words1 = text1.split()
    words2 = text2.split()
    lower_words1 = [w.lower() for w in words1]
    lower_words2 = [w.lower() for w in words2]

    sm = difflib.SequenceMatcher(None, lower_words1, lower_words2)
    v1_marked, v2_marked = [], []

    for tag, i1, i2, j1, j2 in sm.get_opcodes():
        if tag == 'equal':
            v1_marked.extend(words1[i1:i2])
            v2_marked.extend(words2[j1:j2])
        elif tag == 'insert':
            v2_marked.append(f"<<added:{' '.join(words2[j1:j2])}>>")
        elif tag == 'delete':
            v1_marked.append(f"<<removed:{' '.join(words1[i1:i2])}>>")
        elif tag == 'replace':
            v1_marked.append(f"<<removed:{' '.join(words1[i1:i2])}>>")
            v2_marked.append(f"<<added:{' '.join(words2[j1:j2])}>>")

    return " ".join(v1_marked), " ".join(v2_marked)

# ------------------ Main Comparison Logic ------------------ #
def extract_pdf_sections_to_df(pdf_file) -> pd.DataFrame:
    text = extract_text_from_pdf(pdf_file)
    sections = extract_sections_generic(text)
    df = pd.DataFrame(sections)
    df["Clause_name"] = df["heading"].apply(clean_clause_name)
    df["Clause_key"] = df["Clause_name"].apply(clause_join_key)
    return df

def compare_two_pdfs(template_pdf: str, agreement_pdf: str, output_csv: str):
    df1 = extract_pdf_sections_to_df(template_pdf)
    df2 = extract_pdf_sections_to_df(agreement_pdf)

    merged = pd.merge(df1, df2, on="Clause_key", how="outer", suffixes=('_template', '_agreement'))
    merged = merged.fillna("")

    results = []
    for _, row in merged.iterrows():
        key = row["Clause_key"]
        h1, h2 = row["Clause_name_template"], row["Clause_name_agreement"]
        c1, c2 = row["content_template"], row["content_agreement"]

        if c1 and c2:
            if c1.strip().lower() == c2.strip().lower():
                change_type = "Unchanged"
                v1_marked, v2_marked = c1, c2
            else:
                change_type = "Modified"
                v1_marked, v2_marked = mark_diff(c1, c2)
        elif c1 and not c2:
            change_type = "Removed"
            v1_marked, v2_marked = c1, ""
        elif c2 and not c1:
            change_type = "Added"
            v1_marked, v2_marked = "", c2
        else:
            change_type = "Empty"
            v1_marked, v2_marked = "", ""

        results.append({
            "Clause_key": key,
            "Clause_name_template": h1,
            "Clause_content_template": c1,
            "Clause_name_agreement": h2,
            "Clause_content_agreement": c2,
            "change_type": change_type,
            "template_marked": v1_marked,
            "agreement_marked": v2_marked
        })

    out_df = pd.DataFrame(results)
    out_df.to_csv(output_csv, index=False)
    print(f"[INFO] Comparison CSV generated: {output_csv}")

# ------------------ Run Example ------------------ #
if __name__ == "__main__":
    template_pdf = "Old_MSA.pdf"       # Replace with your template path
    agreement_pdf = "New_MSA.pdf"      # Replace with your agreement path
    output_csv = "comparison_result.csv"

    compare_two_pdfs(template_pdf, agreement_pdf, output_csv)
