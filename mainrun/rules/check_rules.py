import argparse, hashlib, json, pathlib, sys
ROOT = pathlib.Path(__file__).resolve().parents[2]  # Go up 2 levels from mainrun/rules/check_rules.py
EVAL_SHA_FILE = ROOT / "rules" / "EVALUATE_SHA256"
EXPECTED = ROOT / "rules" / "EXPECTED_BASELINE.json"  # set after baseline

def sha256(p: pathlib.Path) -> str: return hashlib.sha256(p.read_bytes()).hexdigest()
def die(msg: str): print(f"[RULE VIOLATION] {msg}", file=sys.stderr); sys.exit(1)

def check_eval_hash():
    if not EVAL_SHA_FILE.exists(): die("Missing rules/EVALUATE_SHA256. Snapshot evaluate() first.")
    exp = EVAL_SHA_FILE.read_text().strip()
    
    # Extract just the evaluate() function and hash it (not the entire file)
    import re
    train_py = ROOT / "mainrun" / "train.py"
    content = train_py.read_text()
    pattern = r'def evaluate\(\):\s*\n((?:\s{4}.*\n)*)'
    match = re.search(pattern, content)
    if not match:
        die("evaluate() function not found in train.py")
    
    func_source = match.group(0)
    got = hashlib.sha256(func_source.encode('utf-8')).hexdigest()
    if got != exp: die("train.py has been modified (evaluate() function changed).")

def check_constants(seed: int, epochs: int, val_fraction: float):
    if not EXPECTED.exists(): die("Missing EXPECTED_BASELINE.json. Set it after baseline.")
    exp = json.loads(EXPECTED.read_text())
    if epochs != 7: die(f"epochs must be 7 (got {epochs})")
    if seed != exp["seed"]: die(f"seed must remain {exp['seed']} (got {seed})")
    if abs(val_fraction - exp["val_fraction"]) > 1e-9:
        die(f"val_fraction must remain {exp['val_fraction']} (got {val_fraction})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--precommit", action="store_true")
    ap.add_argument("--payload", help="JSON {train_cfg:{seed,epochs,val_fraction}}")
    args = ap.parse_args()
    check_eval_hash()
    if args.precommit:
        # quick pass; full check runs with real payload during training
        check_constants(seed=1337, epochs=7, val_fraction=0.1); return
    if not args.payload: die("No payload. train.py must send config JSON.")
    data = json.loads(args.payload); tc = data["train_cfg"]
    check_constants(tc["seed"], tc["epochs"], tc["val_fraction"])

if __name__ == "__main__": main()