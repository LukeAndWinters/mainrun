#!/usr/bin/env python3
"""
Build a simple PDF report by concatenating markdown sections for each experiment.
This script appends sequentially; it will not overwrite existing history.
Pure-Python implementation using reportlab to stay Docker-friendly.

Usage:
  python3 scripts/build_report.py --src mainrun/report.md --out mainrun/report.pdf
"""
import argparse, pathlib, re
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.pdfgen import canvas


def draw_wrapped_text(c: canvas.Canvas, text: str, x: float, y: float, max_width: float, line_height: float = 14):
    words = text.split()
    line = []
    cursor_y = y
    for word in words:
        test_line = (" ".join(line + [word])).strip()
        if c.stringWidth(test_line, "Helvetica", 11) <= max_width:
            line.append(word)
        else:
            c.drawString(x, cursor_y, " ".join(line))
            cursor_y -= line_height
            line = [word]
    if line:
        c.drawString(x, cursor_y, " ".join(line))
        cursor_y -= line_height
    return cursor_y


def parse_markdown(md_path: pathlib.Path):
    text_blocks = []
    image_paths = []
    code_block = False
    current_text = []
    for line in md_path.read_text(encoding="utf-8").splitlines():
        if line.strip().startswith("```"):
            code_block = not code_block
            continue
        if code_block:
            current_text.append(line)
            continue
        m = re.search(r"!\[[^\]]*\]\(([^\)]+)\)", line)
        if m:
            if current_text:
                text_blocks.append("\n".join(current_text).strip())
                current_text = []
            image_paths.append(pathlib.Path(m.group(1)))
        else:
            current_text.append(line)
    if current_text:
        text_blocks.append("\n".join(current_text).strip())
    return text_blocks, image_paths


def build_pdf(md_src: pathlib.Path, pdf_out: pathlib.Path):
    pdf_out.parent.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(pdf_out), pagesize=A4)
    width, height = A4
    margin = 2 * cm
    x = margin
    y = height - margin
    max_width = width - 2 * margin

    title = f"MainRun Report"
    c.setFont("Helvetica-Bold", 16)
    c.drawString(x, y, title)
    y -= 24

    text_blocks, image_paths = parse_markdown(md_src) if md_src.exists() else (["Report body not found."], [])

    c.setFont("Helvetica", 11)
    for block in text_blocks:
        for paragraph in block.split("\n\n"):
            y = draw_wrapped_text(c, paragraph, x, y, max_width)
            y -= 8
            if y < margin:
                c.showPage(); c.setFont("Helvetica", 11); y = height - margin

    for img in image_paths:
        p = img if img.is_absolute() else (md_src.parent / img)
        if not p.exists():
            continue
        img_width = max_width
        img_height = img_width * 0.56
        if y - img_height < margin:
            c.showPage(); c.setFont("Helvetica", 11); y = height - margin
        c.drawImage(str(p), x, y - img_height, width=img_width, height=img_height, preserveAspectRatio=True, anchor='n')
        y -= img_height + 12

    c.showPage()
    c.save()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default="mainrun/report.md")
    ap.add_argument("--out", default="mainrun/report.pdf")
    args = ap.parse_args()
    build_pdf(pathlib.Path(args.src), pathlib.Path(args.out))


if __name__ == "__main__":
    main()


