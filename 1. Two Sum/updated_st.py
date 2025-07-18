import pdfplumber
import pandas as pd

def extract_sections_from_pdf(pdf_path: str, output_csv: str):
    """
    Extracts preamble and sections (heading + paragraphs) from a PDF and saves to CSV.
    """
    all_elements = []
    
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            words = page.extract_words(extra_attrs=["size", "fontname", "x0", "top"])
            for w in words:
                all_elements.append(w)

    # Sort by y-coordinate (top) and then x-coordinate for reading order
    all_elements.sort(key=lambda x: (x["top"], x["x0"]))

    sections = []
    current_heading = "Document Preamble"
    current_content = []

    # Detect heading threshold: find avg font size
    font_sizes = [float(w["size"]) for w in all_elements]
    avg_font_size = sum(font_sizes) / len(font_sizes)
    heading_threshold = avg_font_size + 1  # anything bigger than avg is considered heading

    for word in all_elements:
        text = word["text"].strip()
        font_size = float(word["size"])
        
        if font_size >= heading_threshold or text.isupper():
            # Save previous section
            if current_content:
                sections.append({
                    "Heading": current_heading,
                    "Paragraph": " ".join(current_content).strip()
                })
                current_content = []
            current_heading = text  # new heading
        else:
            current_content.append(text)

    # Add last section
    if current_content:
        sections.append({
            "Heading": current_heading,
            "Paragraph": " ".join(current_content).strip()
        })

    # Save to CSV
    df = pd.DataFrame(sections)
    df.to_csv(output_csv, index=False)
    print(f"✅ Extracted {len(sections)} sections into {output_csv}")

# Example usage
extract_sections_from_pdf("Old_MSA.pdf", "sections_extracted.csv")
