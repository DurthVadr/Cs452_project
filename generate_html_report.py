"""
Generate an HTML report from the feature importance comparison markdown file.
Embeds images directly in the HTML for easy viewing.
"""

import os
import base64
import markdown
import re
from pathlib import Path

# Ensure the html_reports directory exists
os.makedirs('html_reports', exist_ok=True)

def read_markdown_file(file_path):
    """Read a markdown file and return its content."""
    with open(file_path, 'r') as f:
        return f.read()

def embed_image(image_path):
    """Embed an image as base64 in HTML."""
    try:
        with open(image_path, 'rb') as img_file:
            img_data = base64.b64encode(img_file.read()).decode('utf-8')
            img_ext = os.path.splitext(image_path)[1][1:]  # Get extension without dot
            return f"data:image/{img_ext};base64,{img_data}"
    except Exception as e:
        print(f"Error embedding image {image_path}: {e}")
        return None

def replace_image_links(html_content):
    """Replace markdown image links with embedded base64 images."""
    # Find all image references in the HTML
    img_pattern = r'<img src="([^"]+)" alt="([^"]+)"'
    
    def replace_match(match):
        img_path = match.group(1)
        alt_text = match.group(2)
        
        # If it's already a data URI, leave it alone
        if img_path.startswith('data:'):
            return match.group(0)
        
        # Otherwise, try to embed the image
        embedded_img = embed_image(img_path)
        if embedded_img:
            return f'<img src="{embedded_img}" alt="{alt_text}"'
        else:
            return match.group(0)
    
    return re.sub(img_pattern, replace_match, html_content)

def add_css_styling(html_content):
    """Add CSS styling to the HTML report."""
    css = """
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f9f9f9;
        }
        h1, h2, h3 {
            color: #2c3e50;
        }
        h1 {
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        h2 {
            border-bottom: 1px solid #bdc3c7;
            padding-bottom: 5px;
            margin-top: 30px;
        }
        img {
            max-width: 100%;
            height: auto;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 5px;
            margin: 20px 0;
            box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        }
        p {
            margin: 15px 0;
        }
        strong {
            color: #2c3e50;
        }
        ul, ol {
            margin: 15px 0;
            padding-left: 30px;
        }
        li {
            margin-bottom: 10px;
        }
        .container {
            background-color: white;
            padding: 30px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
        .image-container {
            text-align: center;
            margin: 30px 0;
        }
        .image-caption {
            font-style: italic;
            color: #7f8c8d;
            margin-top: 10px;
        }
        .key-insight {
            background-color: #f1f9ff;
            border-left: 4px solid #3498db;
            padding: 15px;
            margin: 20px 0;
        }
        .recommendation {
            background-color: #f0fff0;
            border-left: 4px solid #2ecc71;
            padding: 15px;
            margin: 20px 0;
        }
    </style>
    """
    
    # Add the CSS to the head section
    if '<head>' in html_content:
        html_content = html_content.replace('<head>', f'<head>{css}')
    else:
        html_content = f'<html><head>{css}</head><body><div class="container">{html_content}</div></body></html>'
    
    return html_content

def enhance_html_content(html_content):
    """Enhance the HTML content with additional styling and structure."""
    # Wrap images in container divs
    html_content = re.sub(
        r'<p><img src="([^"]+)" alt="([^"]+)"></p>',
        r'<div class="image-container"><img src="\1" alt="\2"><p class="image-caption">\2</p></div>',
        html_content
    )
    
    # Add key insight styling
    html_content = re.sub(
        r'<p><strong>Key (Observations|Insights):</strong></p>',
        r'<div class="key-insight"><h4>Key \1:</h4>',
        html_content
    )
    html_content = re.sub(
        r'(<div class="key-insight">.*?)<h2',
        r'\1</div><h2',
        html_content,
        flags=re.DOTALL
    )
    
    # Add recommendation styling
    html_content = re.sub(
        r'<ol>\s*<li><strong>(.*?)</strong>',
        r'<ol class="recommendation"><li><strong>\1</strong>',
        html_content
    )
    
    return html_content

def generate_html_report(md_file_path, output_path):
    """Generate an HTML report from a markdown file."""
    # Read markdown content
    md_content = read_markdown_file(md_file_path)
    
    # Convert markdown to HTML
    html_content = markdown.markdown(md_content)
    
    # Replace image links with embedded images
    html_content = replace_image_links(html_content)
    
    # Enhance HTML content
    html_content = enhance_html_content(html_content)
    
    # Add CSS styling
    html_content = add_css_styling(html_content)
    
    # Write HTML to file
    with open(output_path, 'w') as f:
        f.write(html_content)
    
    print(f"HTML report generated: {output_path}")

if __name__ == "__main__":
    md_file = "feature_importance_comparison_report.md"
    html_file = "html_reports/feature_importance_comparison.html"
    
    generate_html_report(md_file, html_file)
