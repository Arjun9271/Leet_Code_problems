import pdfplumber
import pandas as pd
import re

def extract_sections_fallback(pdf_path, output_csv):
    with pdfplumber.open(pdf_path) as pdf:
        full_text = ""
        for page in pdf.pages:
            full_text += page.extract_text() + "\n"

    # Split into lines
    lines = full_text.split("\n")
    
    sections = []
    current_heading = "Document Preamble"
    current_content = []

    heading_pattern = re.compile(r'^[A-Z\s\-:&()0-9]+$')  # ALL CAPS lines
    keywords = ["EXHIBIT", "SERVICES", "TERMS", "AGREEMENT", "PAYMENT", "NON-CIRCUMVENTION", "NON-SOLICITATION"]

    for line in lines:
        clean_line = line.strip()
        if not clean_line:
            continue

        # Check if heading
        if heading_pattern.match(clean_line) or clean_line.endswith(":") or any(k in clean_line for k in keywords):
            # Save previous section
            if current_content:
                sections.append({"Heading": current_heading, "Paragraph": " ".join(current_content).strip()})
                current_content = []
            current_heading = clean_line
        else:
            current_content.append(clean_line)

    # Save last section
    if current_content:
        sections.append({"Heading": current_heading, "Paragraph": " ".join(current_content).strip()})

    # Save to CSV
    df = pd.DataFrame(sections)
    df.to_csv(output_csv, index=False)
    print(f"✅ Extracted {len(sections)} sections into {output_csv}")

# Example usage
extract_sections_fallback("Exhibit_B.pdf", "exhibit_sections.csv")
