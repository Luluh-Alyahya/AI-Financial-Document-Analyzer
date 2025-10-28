import os
from pdf2image import convert_from_path
from tqdm import tqdm

# Folders setup
RAW_DIR = "Data/raw_pdfs"
OUT_DIR = "Data/images"
DPI = 300  # image quality (dots per inch)

os.makedirs(OUT_DIR, exist_ok=True)

# Get all PDFs in the folder
pdf_files = [f for f in os.listdir(RAW_DIR) if f.endswith(".pdf")]

# Convert each PDF
for pdf_name in tqdm(pdf_files, desc="Converting PDFs"):
    pdf_path = os.path.join(RAW_DIR, pdf_name)
    company_folder = os.path.join(OUT_DIR, pdf_name.replace(".pdf", ""))
    os.makedirs(company_folder, exist_ok=True)

    try:
        pages = convert_from_path(
            pdf_path,
            dpi=DPI,
            poppler_path=r"C:\poppler-25.07.0\Library\bin"
        )
        for i, page in enumerate(pages, start=1):
            img_path = os.path.join(company_folder, f"page_{i:03d}.png")
            page.save(img_path, "PNG")

        print(f"{pdf_name} | {len(pages)} pages successfully converted.")
    except Exception as e:
        print(f"Error processing {pdf_name}: {e}")
