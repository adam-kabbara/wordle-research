import os
from pathlib import Path
from PyPDF2 import PdfReader, PdfWriter
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
import io
import time
import tqdm

def create_title_page(text, filename):
    packet = io.BytesIO()
    c = canvas.Canvas(packet)
    c.setPageSize((550, 50))

    # Set up styles
    c.setFont("Helvetica-Bold", 12)
    c.setFillColor(colors.darkblue)
    
    # Draw the text
    text_object = c.beginText()
    text_object.setTextOrigin(0, 20)
    
    # Split text into lines and draw each line
    for line in text.split('\n'):
        text_object.textLine(line.strip())
    
    c.drawText(text_object)
    c.save()
    
    # Move to the beginning of the StringIO buffer
    packet.seek(0)
    return PdfReader(packet)

def combine_pdfs(root_dir, output_path):
    pdf_writer = PdfWriter()
    count = 0
    root_path = Path(root_dir)

    # Create a list to store all PDF files with their full paths
    pdf_files = []
    
    # Walk through directory tree and collect all PDF files
    for current_dir, _, files in os.walk(root_dir):
        current_path = Path(current_dir)
        
        # Process PDF files in current directory
        for file in files:
            if file.lower().endswith('.pdf'):
                file_path = current_path / file
                # Store relative path for cleaner display
                rel_path = file_path.relative_to(root_path)
                pdf_files.append((rel_path, file_path))
    
    # Sort files by their relative path
    pdf_files.sort(key=lambda x: str(x[0]))
    
    # Process sorted files
    for rel_path, file_path in tqdm.tqdm(pdf_files):
        # Create title with full path
        title_text = f"File {count}: {rel_path}"
        title_page = create_title_page(title_text, "title.pdf")
        pdf_writer.add_page(title_page.pages[0])
        count += 1
        time.sleep(0.5) # Add a delay to prevent too many open files and make sure reportlab doesnt miss any pages
        
        # Add PDF contents
        try:
            pdf = PdfReader(str(file_path))
            for page in pdf.pages:
                pdf_writer.add_page(page)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
    
    # Save the combined PDF
    with open(output_path, 'wb') as output_file:
        pdf_writer.write(output_file)
    
    return output_path

if __name__ == "__main__":
    # Replace this with your root directory path
    root_directory = r"C:\Users\adamk\Documents\wordle_research\wordle-research\data_analysis\generated_data\paireise (adam framework.py)\data"
    output_file = r"data_analysis\generated_data\master_graph_viewer.pdf"
    output_file = combine_pdfs(root_directory, output_file)
    print(f"Combined PDF saved to: {output_file}")

    # count number of pdf files in dir
    pdf_count = 0
    for current_dir, _, files in os.walk(root_directory):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_count += 1 
    print(f"Number of PDF files processed: {pdf_count}")