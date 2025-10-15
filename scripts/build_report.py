#!/usr/bin/env python3
"""
Build a PDF report that preserves the exact formatting of the markdown preview.
Uses reportlab with proper styling to match markdown rendering.
"""
import argparse
import pathlib
import re
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_JUSTIFY
from reportlab.lib.colors import black, darkblue, darkgreen
from reportlab.lib import colors
import markdown

def create_styles():
    """Create custom styles that match markdown rendering."""
    styles = getSampleStyleSheet()
    
    # Title style
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=24,
        spaceAfter=30,
        alignment=TA_CENTER,
        textColor=black,
        fontName='Helvetica-Bold'
    )
    
    # H1 style
    h1_style = ParagraphStyle(
        'CustomH1',
        parent=styles['Heading1'],
        fontSize=18,
        spaceBefore=20,
        spaceAfter=12,
        textColor=darkblue,
        fontName='Helvetica-Bold'
    )
    
    # H2 style
    h2_style = ParagraphStyle(
        'CustomH2',
        parent=styles['Heading2'],
        fontSize=16,
        spaceBefore=16,
        spaceAfter=8,
        textColor=darkgreen,
        fontName='Helvetica-Bold'
    )
    
    # H3 style
    h3_style = ParagraphStyle(
        'CustomH3',
        parent=styles['Heading3'],
        fontSize=14,
        spaceBefore=12,
        spaceAfter=6,
        textColor=black,
        fontName='Helvetica-Bold'
    )
    
    # Normal text style
    normal_style = ParagraphStyle(
        'CustomNormal',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=6,
        alignment=TA_JUSTIFY,
        fontName='Helvetica'
    )
    
    # Bold text style
    bold_style = ParagraphStyle(
        'CustomBold',
        parent=styles['Normal'],
        fontSize=11,
        spaceAfter=6,
        fontName='Helvetica-Bold'
    )
    
    # Code style
    code_style = ParagraphStyle(
        'CustomCode',
        parent=styles['Code'],
        fontSize=10,
        spaceAfter=6,
        fontName='Courier',
        leftIndent=20,
        rightIndent=20,
        backColor=colors.lightgrey
    )
    
    return {
        'title': title_style,
        'h1': h1_style,
        'h2': h2_style,
        'h3': h3_style,
        'normal': normal_style,
        'bold': bold_style,
        'code': code_style
    }

def process_markdown_content(md_content: str) -> list:
    """Process markdown content and return a list of reportlab elements."""
    styles = create_styles()
    elements = []
    
    # Split content into lines
    lines = md_content.split('\n')
    i = 0
    
    while i < len(lines):
        line = lines[i].strip()
        
        # Skip empty lines
        if not line:
            i += 1
            continue
            
        # Title (first line)
        if i == 0 and line.startswith('# '):
            elements.append(Paragraph(line[2:], styles['title']))
            elements.append(Spacer(1, 0.2 * inch))
            
        # H1 headers
        elif line.startswith('## '):
            elements.append(Paragraph(line[3:], styles['h1']))
            elements.append(Spacer(1, 0.1 * inch))
            
        # H2 headers
        elif line.startswith('### '):
            elements.append(Paragraph(line[4:], styles['h2']))
            elements.append(Spacer(1, 0.05 * inch))
            
        # H3 headers
        elif line.startswith('#### '):
            elements.append(Paragraph(line[5:], styles['h3']))
            elements.append(Spacer(1, 0.03 * inch))
            
        # Images
        elif line.startswith('!['):
            # Extract image path
            match = re.search(r'!\[.*?\]\((.*?)\)', line)
            if match:
                img_path = match.group(1)
                # Convert relative path to absolute
                if img_path.startswith('../'):
                    # Remove ../ prefix and make absolute
                    img_path = img_path[3:]
                elif not img_path.startswith('/'):
                    # If it's already relative, keep as is
                    pass
                
                # Try to find the image file
                full_img_path = None
                possible_paths = [
                    pathlib.Path(img_path),
                    pathlib.Path(f"docs/figures/{pathlib.Path(img_path).name}"),
                    pathlib.Path(f"mainrun/{img_path}"),
                    pathlib.Path(f"../{img_path}")
                ]
                
                for test_path in possible_paths:
                    if test_path.exists():
                        full_img_path = str(test_path)
                        break
                
                if full_img_path:
                    try:
                        # Add image with proper sizing
                        img = Image(full_img_path)
                        # Scale to fit page width
                        img_width = 6 * inch
                        img_height = img.drawHeight * img_width / img.drawWidth
                        img.drawWidth = img_width
                        img.drawHeight = img_height
                        elements.append(img)
                        elements.append(Spacer(1, 0.1 * inch))
                    except Exception as e:
                        # If image can't be loaded, add a placeholder
                        elements.append(Paragraph(f"<i>[Image load error: {full_img_path}]</i>", styles['normal']))
                        elements.append(Spacer(1, 0.05 * inch))
                else:
                    # If image not found, add a placeholder
                    elements.append(Paragraph(f"<i>[Image not found: {img_path}]</i>", styles['normal']))
                    elements.append(Spacer(1, 0.05 * inch))
        
        # Bullet points
        elif line.startswith('- '):
            # Collect all bullet points in this group
            bullet_lines = []
            while i < len(lines) and lines[i].strip().startswith('- '):
                bullet_lines.append(lines[i].strip()[2:])  # Remove '- '
                i += 1
            i -= 1  # Back up one line
            
            # Format bullet points
            for bullet in bullet_lines:
                # Handle bold text in bullets
                bullet_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', bullet)
                elements.append(Paragraph(f"• {bullet_text}", styles['normal']))
            elements.append(Spacer(1, 0.05 * inch))
            
        # Code blocks
        elif line.startswith('```'):
            # Collect code block
            code_lines = []
            i += 1
            while i < len(lines) and not lines[i].strip().startswith('```'):
                code_lines.append(lines[i])
                i += 1
            if code_lines:
                code_text = '\n'.join(code_lines)
                elements.append(Paragraph(f"<font name='Courier'>{code_text}</font>", styles['code']))
                elements.append(Spacer(1, 0.05 * inch))
        
        # Regular paragraphs
        else:
            # Collect paragraph lines
            para_lines = []
            while i < len(lines) and lines[i].strip() and not lines[i].startswith('#') and not lines[i].startswith('!') and not lines[i].startswith('- ') and not lines[i].startswith('```'):
                para_lines.append(lines[i])
                i += 1
            i -= 1  # Back up one line
            
            if para_lines:
                para_text = ' '.join(para_lines)
                # Handle bold text
                para_text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', para_text)
                # Handle inline code
                para_text = re.sub(r'`(.*?)`', r'<font name="Courier">\1</font>', para_text)
                elements.append(Paragraph(para_text, styles['normal']))
                elements.append(Spacer(1, 0.05 * inch))
        
        i += 1
    
    return elements

def build_pdf_from_markdown(markdown_path: pathlib.Path, output_pdf_path: pathlib.Path):
    """Build PDF from markdown file with proper formatting."""
    # Read markdown content
    md_content = markdown_path.read_text(encoding='utf-8')
    
    # Process content
    elements = process_markdown_content(md_content)
    
    # Create PDF
    doc = SimpleDocTemplate(str(output_pdf_path), pagesize=A4, 
                          rightMargin=72, leftMargin=72, 
                          topMargin=72, bottomMargin=18)
    
    # Build PDF
    doc.build(elements)

def main():
    parser = argparse.ArgumentParser(description="Build PDF report from markdown")
    parser.add_argument("--src", default="mainrun/report.md", help="Input markdown file")
    parser.add_argument("--out", default="mainrun/report.pdf", help="Output PDF file")
    args = parser.parse_args()
    
    markdown_path = pathlib.Path(args.src)
    output_path = pathlib.Path(args.out)
    
    if not markdown_path.exists():
        print(f"Error: Markdown file {markdown_path} not found")
        return
    
    try:
        build_pdf_from_markdown(markdown_path, output_path)
        print(f"PDF report generated: {output_path}")
    except Exception as e:
        print(f"Error generating PDF: {e}")

if __name__ == "__main__":
    main()