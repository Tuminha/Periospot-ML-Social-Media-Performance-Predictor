#!/usr/bin/env python3
"""
Export all notebook outputs to a text file
"""

import json
from pathlib import Path
from datetime import datetime

# Paths
NOTEBOOK_PATH = Path(__file__).parent.parent / "notebooks" / "01_post_performance_binary.ipynb"
OUTPUT_PATH = Path(__file__).parent.parent / "artifacts" / f"notebook_outputs_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"

# Ensure artifacts directory exists
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

# Load notebook
print(f"📖 Loading notebook: {NOTEBOOK_PATH}")
with open(NOTEBOOK_PATH, 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# Extract outputs
outputs = []
outputs.append("="*80)
outputs.append("PERIOSPOT ML SOCIAL MEDIA PERFORMANCE PREDICTOR")
outputs.append("Notebook Outputs Export")
outputs.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
outputs.append("="*80)
outputs.append("")

cell_count = 0
output_count = 0

for i, cell in enumerate(notebook['cells']):
    # Process code cells with outputs
    if cell['cell_type'] == 'code' and 'outputs' in cell and cell['outputs']:
        cell_count += 1
        
        # Get the cell source (code)
        code = ''.join(cell['source']).strip()
        
        # Skip empty cells
        if not code:
            continue
            
        outputs.append(f"\n{'='*80}")
        outputs.append(f"Cell {i}")
        outputs.append('='*80)
        outputs.append(f"\nCode:\n{code}\n")
        outputs.append("\n" + "-"*80)
        outputs.append("Output:")
        outputs.append("-"*80 + "\n")
        
        # Extract all output types
        for output in cell['outputs']:
            output_count += 1
            
            # Text output (print statements)
            if 'text' in output:
                outputs.append(''.join(output['text']))
            
            # Data output (from expressions)
            elif 'data' in output:
                # Plain text representation
                if 'text/plain' in output['data']:
                    outputs.append(''.join(output['data']['text/plain']))
                # HTML output
                if 'text/html' in output['data']:
                    outputs.append("[HTML OUTPUT - see notebook for visualization]")
                # Image output
                if 'image/png' in output['data']:
                    outputs.append("[PNG IMAGE OUTPUT - see notebook for visualization]")
                if 'image/jpeg' in output['data']:
                    outputs.append("[JPEG IMAGE OUTPUT - see notebook for visualization]")
                # Plotly/other interactive
                if 'application/vnd.plotly.v1+json' in output['data']:
                    outputs.append("[INTERACTIVE PLOTLY CHART - see notebook]")
            
            # Standard output stream
            elif output.get('output_type') == 'stream':
                if output.get('name') == 'stdout':
                    outputs.append(''.join(output.get('text', [])))
                elif output.get('name') == 'stderr':
                    outputs.append('STDERR:\n' + ''.join(output.get('text', [])))
            
            # Errors
            elif output.get('output_type') == 'error':
                outputs.append(f"\nERROR: {output.get('ename', 'Unknown')}: {output.get('evalue', '')}")
                if 'traceback' in output:
                    # Clean ANSI escape codes from traceback
                    import re
                    traceback = ''.join(output['traceback'])
                    traceback = re.sub(r'\x1b\[[0-9;]*m', '', traceback)  # Remove ANSI codes
                    outputs.append(traceback)

# Add summary
outputs.append("\n\n" + "="*80)
outputs.append("EXPORT SUMMARY")
outputs.append("="*80)
outputs.append(f"Total cells with output: {cell_count}")
outputs.append(f"Total outputs extracted: {output_count}")
outputs.append(f"Export completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
outputs.append("="*80)

# Write to file
with open(OUTPUT_PATH, 'w', encoding='utf-8') as f:
    f.write('\n'.join(outputs))

print(f"✅ Outputs exported successfully!")
print(f"   File: {OUTPUT_PATH}")
print(f"   Cells with output: {cell_count}")
print(f"   Total outputs: {output_count}")
print(f"   File size: {OUTPUT_PATH.stat().st_size / 1024:.1f} KB")

