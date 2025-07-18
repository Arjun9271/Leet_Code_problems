import os
import re
import pdfplumber
import pandas as pd

def extract_text_from_pdf(pdf_path: str) -> str:
    """Extracts all text from a PDF file."""
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text(x_tolerance=2, y_tolerance=2)
            if page_text:
                text += page_text + "\n"
    return text

def extract_sections_generic(text: str):
    """Detects headings and groups following paragraphs under them."""
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    
    sections = []
    current_heading = "Document Preamble"
    current_content = []

    # Heading detection logic
    def is_heading(line: str) -> bool:
        if len(line) < 3: 
            return False
        if line.endswith(":") and sum(c.isalpha() for c in line) >= 3:
            return True
        if line.isupper() and len(line.split()) <= 12:
            return True
        if re.match(r'^(exhibit|nature of|services|terms|payment|non[- ]|agreement|definitions)', line, re.I):
            return True
        if re.match(r'^section\s+\d+', line, re.I):
            return True
        return False

    for ln in lines:
        if is_heading(ln):
            if current_content:
                sections.append({"Clause_name": current_heading, "Clause_content": " ".join(current_content).strip()})
                current_content = []
            current_heading = ln
        else:
            current_content.append(ln)

    # Append last section
    if current_content:
        sections.append({"Clause_name": current_heading, "Clause_content": " ".join(current_content).strip()})

    return sections

def extract_pdf_to_csv(pdf_path: str, output_csv: str):
    """Extracts headings + content from PDF and saves to CSV."""
    text = extract_text_from_pdf(pdf_path)
    sections = extract_sections_generic(text)
    df = pd.DataFrame(sections)
    df.to_csv(output_csv, index=False)
    print(f"[INFO] Extracted {len(df)} sections -> {output_csv}")

# Example Usage
if __name__ == "__main__":
    pdf_file = "YourDocument.pdf"  # Replace with your PDF
    output_csv = "extracted_sections.csv"
    extract_pdf_to_csv(pdf_file, output_csv)
