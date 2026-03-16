"""
Convert Markdown files to PDF using xhtml2pdf.
Usage: python convert_to_pdf.py
"""

import os
import re
import base64
import markdown
from xhtml2pdf import pisa

DOCS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)))

FILES = [
    ("architecture.md", "architecture.pdf"),
    ("mentor_report.md", "mentor_report.pdf"),
]

CSS = """
<style>
  @page {
    margin: 2cm 2.5cm;
    size: A4;
  }
  body {
    font-family: "Helvetica Neue", Helvetica, Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #1a1a1a;
  }
  h1 {
    font-size: 22pt;
    font-weight: 700;
    color: #0d1b2a;
    border-bottom: 2px solid #2563EB;
    padding-bottom: 6pt;
    margin-top: 0;
  }
  h2 {
    font-size: 15pt;
    font-weight: 700;
    color: #1e3a5f;
    border-bottom: 1px solid #93c5fd;
    padding-bottom: 3pt;
    margin-top: 18pt;
  }
  h3 {
    font-size: 12pt;
    font-weight: 700;
    color: #1d4ed8;
    margin-top: 12pt;
  }
  h4 {
    font-size: 11pt;
    font-weight: 700;
    color: #333;
    margin-top: 10pt;
  }
  p { margin: 6pt 0; }
  code {
    font-family: "Courier New", monospace;
    font-size: 9pt;
    background-color: #f1f5f9;
    padding: 1pt 3pt;
    border-radius: 3pt;
    color: #c0392b;
  }
  pre {
    background-color: #f8fafc;
    border: 1px solid #cbd5e1;
    border-left: 4px solid #2563EB;
    padding: 10pt;
    font-size: 8.5pt;
    overflow: hidden;
    margin: 8pt 0;
  }
  pre code {
    background: none;
    color: #1a1a1a;
    padding: 0;
  }
  blockquote {
    border-left: 4px solid #93c5fd;
    background: #eff6ff;
    margin: 8pt 0;
    padding: 6pt 12pt;
    color: #1e40af;
  }
  table {
    border-collapse: collapse;
    width: 100%;
    margin: 10pt 0;
    font-size: 10pt;
  }
  th {
    background-color: #1e3a5f;
    color: white;
    padding: 6pt 8pt;
    text-align: left;
    font-weight: 700;
  }
  td {
    padding: 5pt 8pt;
    border: 1px solid #cbd5e1;
  }
  tr:nth-child(even) td { background-color: #f8fafc; }
  ul, ol {
    margin: 6pt 0;
    padding-left: 18pt;
  }
  li { margin: 3pt 0; }
  img {
    max-width: 100%;
    height: auto;
    display: block;
    margin: 10pt auto;
  }
  strong { color: #0f172a; }
  em { color: #475569; }
  hr {
    border: none;
    border-top: 1px solid #cbd5e1;
    margin: 14pt 0;
  }
</style>
"""


def image_to_base64(img_path):
    """Read an image and return a base64 data URI."""
    try:
        with open(img_path, "rb") as f:
            data = base64.b64encode(f.read()).decode("ascii")
        ext = os.path.splitext(img_path)[1].lower().lstrip(".")
        mime = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "gif": "image/gif"}.get(ext, "image/png")
        return f"data:{mime};base64,{data}"
    except Exception as e:
        print(f"  WARN: Could not load image {img_path}: {e}")
        return None


def replace_images_with_base64(md_content, docs_dir):
    """Replace all markdown image references with base64 data URIs."""

    def process_line(line):
        # Match ![alt](path) where path may contain parentheses
        # Strategy: find ![ then match to nearest ] then ( then match to .png/.jpg/.gif)
        result = []
        i = 0
        while i < len(line):
            if line[i:i+2] == '![':
                # Find closing ] for alt text
                alt_end = line.find(']', i+2)
                if alt_end == -1:
                    result.append(line[i])
                    i += 1
                    continue
                # Check for opening (
                if alt_end + 1 < len(line) and line[alt_end + 1] == '(':
                    alt = line[i+2:alt_end]
                    # Find the path: look for extension to locate end
                    path_start = alt_end + 2
                    path_end = -1
                    for ext in ['.png', '.jpg', '.jpeg', '.gif']:
                        idx = line.find(ext + ')', path_start)
                        if idx != -1:
                            path_end = idx + len(ext)
                            break
                    if path_end != -1:
                        src = line[path_start:path_end]
                        if src.startswith('http'):
                            result.append(line[i:path_end+1])
                        else:
                            from urllib.parse import unquote
                            src_decoded = unquote(src)
                            abs_path = os.path.normpath(os.path.join(docs_dir, src_decoded))
                            data_uri = image_to_base64(abs_path)
                            if data_uri:
                                result.append(f'<img src="{data_uri}" alt="{alt}" />')
                            else:
                                result.append(f'<p><em>[Image: {alt}]</em></p>')
                        i = path_end + 1  # skip past the closing )
                        continue
            result.append(line[i])
            i += 1
        return ''.join(result)

    lines = md_content.split('\n')
    processed = [process_line(l) if '![' in l else l for l in lines]
    return '\n'.join(processed)


def md_to_pdf(md_path, pdf_path):
    print(f"Converting: {os.path.basename(md_path)}")

    with open(md_path, "r", encoding="utf-8") as f:
        md_content = f.read()

    docs_dir = os.path.dirname(os.path.abspath(md_path))
    md_content = replace_images_with_base64(md_content, docs_dir)

    html_body = markdown.markdown(
        md_content,
        extensions=["tables", "fenced_code", "nl2br", "sane_lists"],
    )

    # Replace emoji for PDF compatibility
    replacements = {
        "\u2705": "[OK]", "\u274c": "[X]", "\U0001f534": "[!]",
        "\U0001f7e1": "[~]", "\U0001f7e2": "[+]", "\u23f9\ufe0f": "[stop]",
        "\u2764\ufe0f": "[heart]", "\U0001f4ca": "[chart]", "\u26a1": "[fast]",
        "\u2744\ufe0f": "[frozen]", "\u2714": "[ok]", "\u2716": "[fail]",
        "\U0001fac0": "[ecg]", "\U0001f3e5": "[hospital]", "\U0001f4f7": "[cam]",
        "\U0001f4e6": "[box]", "\U0001f527": "[tool]", "\U0001f5a5\ufe0f": "[pc]",
        "\U0001f9ea": "[science]", "\U0001f680": "[rocket]", "\u2b50\ufe0f": "[star]",
    }
    for emoji, text in replacements.items():
        html_body = html_body.replace(emoji, text)

    full_html = f"""<!DOCTYPE html>
<html>
<head>
  <meta charset="utf-8">
  {CSS}
</head>
<body>
{html_body}
</body>
</html>"""

    with open(pdf_path, "wb") as pdf_file:
        result = pisa.CreatePDF(
            full_html.encode("utf-8"),
            dest=pdf_file,
            encoding="utf-8",
        )

    if result.err:
        print(f"  ERROR during PDF creation (errors: {result.err})")
        return False
    else:
        size_kb = os.path.getsize(pdf_path) // 1024
        print(f"  DONE: {size_kb} KB -> {os.path.basename(pdf_path)}")
        return True


if __name__ == "__main__":
    success_count = 0
    for md_file, pdf_file in FILES:
        md_path = os.path.join(DOCS_DIR, md_file)
        pdf_path = os.path.join(DOCS_DIR, pdf_file)
        if not os.path.exists(md_path):
            print(f"  SKIP: {md_file} not found")
            continue
        ok = md_to_pdf(md_path, pdf_path)
        if ok:
            success_count += 1

    print(f"\nResult: {success_count}/{len(FILES)} converted.")
    print(f"PDFs saved in: {DOCS_DIR}")
