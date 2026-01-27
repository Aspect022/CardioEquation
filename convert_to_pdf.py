"""
Convert SYSTEM_ARCHITECTURE.md to PDF with embedded images
"""
import os
import markdown2
from xhtml2pdf import pisa
import base64

# Paths
md_path = "docs/SYSTEM_ARCHITECTURE.md"
pdf_path = "docs/SYSTEM_ARCHITECTURE.pdf"
docs_dir = "docs"

# Read markdown
with open(md_path, 'r', encoding='utf-8') as f:
    md_content = f.read()

# Convert markdown to HTML
html_content = markdown2.markdown(md_content, extras=[
    'tables', 
    'fenced-code-blocks', 
    'code-friendly',
    'cuddled-lists',
    'header-ids'
])

# Function to embed images as base64
def embed_images(html, base_dir):
    import re
    
    def replace_img(match):
        src = match.group(1)
        # Handle relative paths
        if src.startswith('./'):
            src = src[2:]
        elif src.startswith('../'):
            src = os.path.join('..', src[3:])
            
        # Construct full path
        full_path = os.path.normpath(os.path.join(base_dir, src))
        
        # URL decode spaces
        full_path = full_path.replace('%20', ' ')
        
        print(f"Looking for image: {full_path}")
        
        if os.path.exists(full_path):
            with open(full_path, 'rb') as img_file:
                img_data = base64.b64encode(img_file.read()).decode('utf-8')
                ext = os.path.splitext(full_path)[1].lower()
                mime = 'image/png' if ext == '.png' else 'image/jpeg'
                return f'src="data:{mime};base64,{img_data}"'
        else:
            print(f"Image not found: {full_path}")
            return match.group(0)
    
    return re.sub(r'src="([^"]+)"', replace_img, html)

# Embed images
html_with_images = embed_images(html_content, docs_dir)

# Create full HTML document with styling
full_html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <style>
        @page {{
            size: A4;
            margin: 1.5cm;
        }}
        body {{
            font-family: Arial, Helvetica, sans-serif;
            font-size: 11pt;
            line-height: 1.4;
            color: #333;
        }}
        h1 {{
            color: #1a5276;
            font-size: 24pt;
            border-bottom: 3px solid #1a5276;
            padding-bottom: 10px;
        }}
        h2 {{
            color: #2874a6;
            font-size: 16pt;
            margin-top: 20px;
            border-bottom: 1px solid #2874a6;
        }}
        h3 {{
            color: #3498db;
            font-size: 13pt;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 15px 0;
            font-size: 10pt;
        }}
        th, td {{
            border: 1px solid #bdc3c7;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #3498db;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        code {{
            background-color: #f4f4f4;
            padding: 2px 5px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 9pt;
        }}
        pre {{
            background-color: #f4f4f4;
            padding: 10px;
            border-radius: 5px;
            overflow-x: auto;
            font-size: 8pt;
            border: 1px solid #ddd;
        }}
        blockquote {{
            border-left: 4px solid #3498db;
            margin: 15px 0;
            padding: 10px 15px;
            background-color: #f8f9fa;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 15px auto;
            border: 1px solid #ddd;
            border-radius: 5px;
        }}
        hr {{
            border: none;
            border-top: 2px solid #e0e0e0;
            margin: 25px 0;
        }}
        .important {{
            background-color: #fff3cd;
            border-left: 4px solid #ffc107;
            padding: 10px;
            margin: 15px 0;
        }}
    </style>
</head>
<body>
{html_with_images}
</body>
</html>
"""

# Convert to PDF
with open(pdf_path, "wb") as pdf_file:
    pisa_status = pisa.CreatePDF(full_html, dest=pdf_file)

if pisa_status.err:
    print(f"Error creating PDF: {pisa_status.err}")
else:
    print(f"✅ PDF created successfully: {pdf_path}")
