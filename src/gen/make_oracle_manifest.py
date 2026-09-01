"""Prompt-contrast manifest: boundary-flank rows duplicated with oracle prompts."""
import json
from pathlib import Path
from gen.bundle_a_spec import ORACLE

ROOT = Path("artifacts/bundle_a")
FLANK_S = (0.95, 1.05)

def main():
    rows = [json.loads(l) for l in (ROOT/"manifest.jsonl").read_text().splitlines() if l.strip()]
    out = []
    for r in rows:
        if r["role"] != "grid" or round(r["S"], 2) not in FLANK_S:
            continue
        for tag, outcome in (("correct", r["outcome"]), ("wrong", 1 - r["outcome"])):
            v = dict(r)
            v["id"] = f'{r["id"]}__{tag}'
            v["prompt"] = ORACLE[r["family"]][outcome]
            v["oracle"] = tag
            v["base_id"] = r["id"]
            out.append(v)
    path = ROOT/"manifest_oracle.jsonl"
    path.write_text("".join(json.dumps(v) + "\n" for v in out))
    print(f"{len(out)} rows -> {path}")
    for v in out:
        print(f'  {v["id"]:28s} S={v["S"]}  prompt={v["prompt"][:60]}...')

if __name__ == "__main__":
    main()
