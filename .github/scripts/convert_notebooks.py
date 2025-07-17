#!/usr/bin/env python3
"""
Convert Databricks .py notebook files to HTML for GitHub Pages publishing.
"""

import os
import re
import markdown
import glob
import html

def parse_databricks_notebook(file_path):
    """Parse a Databricks notebook file and extract cells."""
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    cells = []
    current_cell = {'type': 'code', 'content': ''}
    
    lines = content.split('\n')
    
    for line in lines:
        if line.startswith('# MAGIC %md'):
            # Save current cell if it has content
            if current_cell['content'].strip():
                cells.append(current_cell)
            # Start new markdown cell
            current_cell = {'type': 'markdown', 'content': ''}
        elif line.startswith('# MAGIC'):
            # Continue markdown cell
            markdown_content = line.replace('# MAGIC ', '').replace('# MAGIC', '')
            current_cell['content'] += markdown_content + '\n'
        elif line.startswith('# COMMAND ----------'):
            # Save current cell and start new code cell
            if current_cell['content'].strip():
                cells.append(current_cell)
            current_cell = {'type': 'code', 'content': ''}
        elif line.startswith('# DBTITLE'):
            # Skip DBTITLE lines but start new code cell
            if current_cell['content'].strip():
                cells.append(current_cell)
            current_cell = {'type': 'code', 'content': ''}
        else:
            # Regular code line
            if current_cell['type'] == 'code':
                current_cell['content'] += line + '\n'
    
    # Add final cell
    if current_cell['content'].strip():
        cells.append(current_cell)
    
    return cells

def convert_to_html(notebook_path, output_dir):
    """Convert a single notebook to HTML."""
    cells = parse_databricks_notebook(notebook_path)
    
    # Generate HTML
    notebook_name = os.path.splitext(os.path.basename(notebook_path))[0]
    
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{notebook_name} - Customer Segmentation</title>
    <link href="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/themes/prism.min.css" rel="stylesheet" />
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', system-ui, sans-serif;
            line-height: 1.6;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background: #f8f9fa;
        }}
        .notebook {{
            background: white;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .header {{
            background: linear-gradient(135deg, #FF3621, #FF6B47);
            color: white;
            padding: 30px;
            text-align: center;
        }}
        .cell {{
            border-bottom: 1px solid #e9ecef;
            padding: 20px 30px;
        }}
        .cell:last-child {{
            border-bottom: none;
        }}
        .cell-markdown {{
            background: #fff;
        }}
        .cell-code {{
            background: #f8f9fa;
        }}
        .cell-markdown h1, .cell-markdown h2, .cell-markdown h3 {{
            color: #FF3621;
        }}
        .cell-markdown p {{
            margin: 10px 0;
        }}
        .cell-markdown ul, .cell-markdown ol {{
            margin: 10px 0;
            padding-left: 20px;
        }}
        .cell-markdown blockquote {{
            border-left: 4px solid #FF3621;
            margin: 10px 0;
            padding: 10px 20px;
            background: #f8f9fa;
        }}
        pre {{
            background: #f1f3f4;
            padding: 15px;
            border-radius: 8px;
            overflow-x: auto;
            margin: 10px 0;
        }}
        code {{
            background: #f1f3f4;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: 'Monaco', 'Consolas', monospace;
        }}
        .nav {{
            position: fixed;
            top: 20px;
            right: 20px;
            background: white;
            padding: 10px;
            border-radius: 8px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }}
        .nav a {{
            display: block;
            color: #FF3621;
            text-decoration: none;
            padding: 5px 10px;
            border-radius: 4px;
            margin: 2px 0;
        }}
        .nav a:hover {{
            background: #f8f9fa;
        }}
        .back-btn {{
            background: #FF3621;
            color: white;
            padding: 10px 20px;
            text-decoration: none;
            border-radius: 6px;
            display: inline-block;
            margin-bottom: 20px;
        }}
        .back-btn:hover {{
            background: #e02d1a;
        }}
    </style>
</head>
<body>
    <a href="index.html" class="back-btn">← Back to Overview</a>
    
    <div class="nav">
        <strong>Notebooks:</strong>
        <a href="01_Data_Generation.html">Data Generation</a>
        <a href="02_Segmentation_Analysis.html">Segmentation</a>
        <a href="03_Business_Insights.html">Business Insights</a>
    </div>
    
    <div class="notebook">
        <div class="header">
            <h1>{notebook_name.replace('_', ' ')}</h1>
            <p>Customer Segmentation Solution Accelerator</p>
        </div>
"""
    
    for cell in cells:
        if cell['type'] == 'markdown':
            # Convert markdown to HTML
            md_content = markdown.markdown(cell['content'], extensions=['extra', 'codehilite'])
            html_content += f'        <div class="cell cell-markdown">\n{md_content}\n        </div>\n'
        else:
            # Code cell
            escaped_code = html.escape(cell['content'])
            html_content += f'''        <div class="cell cell-code">
            <pre><code class="language-python">{escaped_code}</code></pre>
        </div>
'''
    
    html_content += """    </div>
    
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/components/prism-core.min.js"></script>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/prism/1.24.1/plugins/autoloader/prism-autoloader.min.js"></script>
</body>
</html>"""
    
    # Write HTML file
    output_file = os.path.join(output_dir, f"{notebook_name}.html")
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Converted {notebook_path} -> {output_file}")

def main():
    """Main conversion script."""
    # Create output directory
    output_dir = "site"
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all .py notebook files
    notebook_files = glob.glob("notebooks/*.py")
    
    if not notebook_files:
        print("No notebook files found in notebooks/ directory")
        return
    
    print(f"Found {len(notebook_files)} notebook files to convert:")
    
    # Convert each notebook
    for notebook_file in sorted(notebook_files):
        convert_to_html(notebook_file, output_dir)
    
    print(f"\\nConversion complete! HTML files saved to {output_dir}/")

if __name__ == "__main__":
    main()