import pdfplumber
import re
import pandas as pd
from typing import List, Dict


# ---------- STEP 1: Load PDF and extract text ----------
def load_pdf_text(file_path: str) -> str:
    """Extract text from a PDF file using pdfplumber."""
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text.strip()


# ---------- STEP 2: Normalize PDF text ----------
def normalize_text(text: str) -> str:
    """Clean up extracted text for easier parsing."""
    text = text.replace("–", "-").replace("—", "-")  # normalize dashes
    text = re.sub(r'[ \t]+', ' ', text)  # collapse extra spaces
    return text.strip()


# ---------- STEP 3: Extract sections ----------
def extract_sections(text: str) -> List[Dict[str, str]]:
    """
    Extract sections from text, including preamble.
    Section pattern: "Section <number> - <title>"
    If no section headings found, entire text is Document Preamble.
    """
    section_pattern = re.compile(
        r'(Section\s+\d+(?:\.\d+)?\s*[-:]\s*[A-Za-z0-9 ,&/().]+)',  # e.g., Section 1 - Payment Terms
        flags=re.IGNORECASE
    )

    matches = list(section_pattern.finditer(text))
    sections = []

    if not matches:
        return [{"section_title": "Document Preamble", "content": text.strip()}]

    # Preamble: everything before first section
    first_start = matches[0].start()
    preamble = text[:first_start].strip()
    if preamble:
        sections.append({"section_title": "Document Preamble", "content": preamble})

    # Each section
    for i, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        content = text[start:end].strip()
        sections.append({"section_title": title, "content": content})

    return sections


# ---------- STEP 4: Save to CSV ----------
def save_sections_to_csv(sections: List[Dict[str, str]], output_path: str):
    """Save section data into a CSV file."""
    df = pd.DataFrame(sections)
    df.to_csv(output_path, index=False)
    print(f"✅ Saved {len(df)} sections to {output_path}")


# ---------- Example Usage ----------
if __name__ == "__main__":
    pdf_file = "Old_MSA.pdf"  # Replace with your PDF path
    output_csv = "document_sections.csv"

    raw_text = load_pdf_text(pdf_file)
    clean_text = normalize_text(raw_text)
    extracted_sections = extract_sections(clean_text)
    save_sections_to_csv(extracted_sections, output_csv)

    print("\nExtracted Sections:")
    for sec in extracted_sections:
        print(f"\n{sec['section_title']}\n{sec['content'][:100]}...")
