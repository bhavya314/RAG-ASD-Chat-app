import fitz  # PyMuPDF
import re
import os
import pytesseract
from pdf2image import convert_from_path

# check if the PDF is image-based or selectable text
def is_image_based_pdf(pdf_path, check_pages=3):
    doc = fitz.open(pdf_path)
    for i in range(min(len(doc), check_pages)):
        text = doc[i].get_text()
        if text.strip():  # If any text is found
            return False
    return True  # No text found in any checked page

# some PDFs have a lot of boilerplate text before the main content starts
# we try to find the main content start by looking for keywords like "Abstract" or "Introduction"
# if not found, we skip the first ~1000 characters (title, authors, etc.)
def find_main_start(text):
    # 1. Try "Abstract"
    match = re.search(r'\babstract\b', text, flags=re.IGNORECASE)
    if match:
        return text[match.start():]

    # 2. Try "Introduction" or variants
    match = re.search(r'\b(1\.\s*)?introduction\b', text, flags=re.IGNORECASE)
    if match:
        return text[match.start():]

    # 3. Fallback: return text after skipping first ~1000 characters (title, authors, etc.)
    return text[1000:]

# if image-based, convert the PDF pages to images and extract text using OCR
def ocr_pdf(pdf_path):
    # Convert PDF pages to images
    images = convert_from_path(pdf_path, dpi=300)
    
    # Run OCR on each page
    full_text = ""
    for i, image in enumerate(images):
        page_text = pytesseract.image_to_string(image)
        full_text += f"\n\n--- Page {i+1} ---\n\n" + page_text
    return full_text

# Remove numeric blocks from the text
def remove_numeric_blocks(text):
    lines = text.split('\n')
    cleaned = []
    numeric_run = 0
    for line in lines:
        digit_ratio = len(re.findall(r'[\d.±%]', line)) / (len(line) + 1e-6)
        if digit_ratio > 0.5 or re.match(r'^\s*[\d.,]+(\s+[\d.,%±]+)*\s*$', line):
            numeric_run += 1
        else:
            numeric_run = 0

        if numeric_run >= 3:
            continue
        if numeric_run == 0:
            cleaned.append(line)
    return "\n".join(cleaned)

# remove tabular blocks from the text
def remove_tabular_blocks(text):
    lines = text.split('\n')
    cleaned = []
    for line in lines:
        # remove lines with multiple spaces/tabs indicating tabular data
        if len(re.findall(r'\s{3,}', line)) >= 1 or line.count('\t') >= 2:
            continue
        cleaned.append(line)
    return "\n".join(cleaned)

# remove footer header blocks from the text
def remove_headers_footers(text):
    text = re.sub(r'\bPage \d+(\s+of \d+)?\b', '', text, flags=re.IGNORECASE)
    text = re.sub(r'NIH-PA Author Manuscript', '', text)
    text = re.sub(r'Author manuscript; available in PMC.*?\n', '', text)
    text = re.sub(r'J Autism Dev Disord.*?\n', '', text)
    text = re.sub(r'bioRxiv.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'medRxiv.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Scientific Reports.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'Copyright.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'All rights reserved.*?\n', '', text, flags=re.IGNORECASE)
    text = re.sub(r'^[\s\.\-—_]{3,}$', '', text, flags=re.MULTILINE)
    return text

# Clean the extracted text by removing references, URLs, figures, and boilerplate text
def clean_text(text):
    # Remove pre-content (use your fallback strategy)
    text = find_main_start(text)

    text = re.sub(r'(?i)\b(references|acknowledgements|appendix)\b.*', '', text, flags=re.DOTALL)

    # Remove citations
    text = re.sub(r'\[\s*[\w\-]+ et al\., \d{4}(;\s*[\w\-]+ et al\., \d{4})*\s*\]', '', text)
    text = re.sub(r'\b\w+ et al\.,? \d{4}\b', '', text)
    text = re.sub(r'\b[A-Z][a-z]+ et al\.\b', '', text)

    # Remove URLs, DOIs, emails
    text = re.sub(r'http\S+|www\.\S+|doi:\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)

    # Remove figure/table captions
    text = re.sub(r'(Table|Figure|Fig\.)\s*\d+.*', '', text)

    # Remove numeric blocks
    text = remove_numeric_blocks(text)

    # Remove tabular data
    text = remove_tabular_blocks(text)

    # Remove headers & footers
    text = remove_headers_footers(text)

    # Collapse duplicate lines
    lines = text.splitlines()
    deduped = []
    prev = ""
    for line in lines:
        if line.strip() != prev.strip():
            deduped.append(line)
        prev = line
    text = "\n".join(deduped)

    # Clean up extra whitespace
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'[ \t]{2,}', ' ', text)

    return text.strip()


# Extract text from a PDF file and clean it
def extract_clean_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text_chunks = []

    for page in doc:
        text = page.get_text()
        text_chunks.append(text)

    full_text = "\n".join(text_chunks)
    return clean_text(full_text)

# Save the cleaned text to a file
def save_clean_text_to_file(text, output_path):
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text)

# processing the selecable text PDFs
def process_pdf(pdf_path, output_dir):
    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    clean_text = extract_clean_text_from_pdf(pdf_path)
    output_path = os.path.join(output_dir, f"{filename}_cleaned.txt")
    save_clean_text_to_file(clean_text, output_path)
    print(f"✅ Saved cleaned text to: {output_path}")

# processing the image-based PDFs using OCR
def process_ocr_pdf(pdf_path, output_dir):
    raw = ocr_pdf(pdf_path)
    clean = clean_text(raw)
    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, f"{filename}_cleaned.txt"), "w", encoding="utf-8") as f:
        f.write(clean)
    print(f"✅ OCR Cleaned: {filename}")



# Example usage
if __name__ == "__main__":
    output_dir = "cleaned_texts"
    for pdf_file in os.listdir("papers/"):
        if pdf_file.endswith(".pdf"):
            pdf_path = os.path.join("papers", pdf_file)
            if is_image_based_pdf(pdf_path):
                process_ocr_pdf(pdf_path, output_dir)
            else:
                process_pdf(pdf_path, output_dir)
                

