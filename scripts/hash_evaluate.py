#!/usr/bin/env python3
"""
Extract evaluate() function from mainrun/train.py and compute SHA256 hash.
WHAT: Hash only the evaluate() function body to detect modifications
WHY: Enforce that evaluate() function remains unchanged per project rules
"""
import hashlib
import re
import sys
from pathlib import Path

def extract_evaluate_function(file_path):
    """Extract the evaluate() function source from train.py"""
    content = Path(file_path).read_text()
    
    # Find the evaluate function definition
    # Look for 'def evaluate():' and capture everything until the next function or end of file
    pattern = r'def evaluate\(\):\s*\n((?:\s{4}.*\n)*)'
    match = re.search(pattern, content)
    
    if not match:
        raise ValueError("evaluate() function not found in train.py")
    
    return match.group(0)

def main():
    train_py = Path("mainrun/train.py")
    if not train_py.exists():
        print("Error: mainrun/train.py not found", file=sys.stderr)
        sys.exit(1)
    
    try:
        func_source = extract_evaluate_function(train_py)
        hash_value = hashlib.sha256(func_source.encode('utf-8')).hexdigest()
        
        # Write to rules/EVALUATE_SHA256
        rules_dir = Path("rules")
        rules_dir.mkdir(exist_ok=True)
        hash_file = rules_dir / "EVALUATE_SHA256"
        hash_file.write_text(hash_value)
        
        print(f"evaluate() function hash: {hash_value}")
        print(f"Written to: {hash_file}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
