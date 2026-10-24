#!/usr/bin/env python3
"""Check a contract (PDF or .txt) against the government templates.

Usage:
    python check_template.py contract.pdf [--templates DIR] [--type works_construction] [--json]

Importable:
    from check_template import check_contract
    report = check_contract(text, templates_dir)
"""
import argparse, glob, json, os, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_DIR = os.path.join(HERE, "templates")
SEV_W = {"high": 3, "medium": 2, "low": 1}


def read_text(path):
    if path.lower().endswith(".pdf"):
        try:
            out = subprocess.run(["pdftotext", "-layout", path, "-"], capture_output=True, text=True, timeout=120)
            if out.stdout.strip():
                return out.stdout
        except Exception:
            pass
        from pypdf import PdfReader
        return "\n".join((p.extract_text() or "") for p in PdfReader(path).pages)
    with open(path, encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_templates(d):
    out = {}
    for p in glob.glob(os.path.join(d, "*.json")):
        if os.path.basename(p) == "index.json":
            continue
        with open(p, encoding="utf-8") as f:
            t = json.load(f)
        out[t["id"]] = t
    return out


def _hits(patterns, text):
    found = []
    for pat in patterns:
        m = re.search(pat, text, re.I)
        if m:
            found.append((pat, m))
    return found


def _snippet(text, m, width=110):
    s, e = max(0, m.start() - width), min(len(text), m.end() + width)
    return re.sub(r"\s+", " ", text[s:e]).strip()


def detect_type(text, templates):
    scores = {}
    for tid, t in templates.items():
        scores[tid] = len(_hits(t["detect_keywords"], text))
    ranked = sorted(scores.items(), key=lambda x: -x[1])
    best, best_s = ranked[0]
    second_s = ranked[1][1] if len(ranked) > 1 else 0
    conf = "high" if best_s >= 5 and best_s - second_s >= 2 else "medium" if best_s >= 3 else "low"
    return best, scores, conf


def check_contract(text, templates_dir=DEFAULT_DIR, force_type=None):
    templates = load_templates(templates_dir)
    if force_type:
        tid, scores, conf = force_type, {}, "user-specified"
    else:
        tid, scores, conf = detect_type(text, templates)
    t = templates[tid]

    present, missing = [], []
    for cl in t["required_clauses"]:
        h = _hits(cl["keywords"], text)
        if h:
            present.append({"id": cl["id"], "title": cl["title"], "severity": cl["severity"],
                            "evidence": _snippet(text, h[0][1]), "matched": len(h),
                            "strength": "strong" if len(h) >= 2 else "single-keyword"})
        else:
            missing.append({"id": cl["id"], "title": cl["title"], "severity": cl["severity"],
                            "guidance": cl.get("guidance", "")})

    flags = []
    for r in t["red_flags"]:
        m = re.search(r["pattern"], text, re.I | re.S)
        if m:
            flags.append({"id": r["id"], "title": r["title"], "why": r["why"], "evidence": _snippet(text, m)})

    bench = []
    for b in t.get("benchmarks", []):
        m = re.search(b["pattern"], text, re.I | re.S)
        item = {"id": b["id"], "label": b["label"], "expected": b["expected"], "verify_against_source": b.get("verify", False)}
        if m:
            item["found_value"] = m.group(1)
            item["evidence"] = _snippet(text, m)
            cap = b.get("cap_pct_max")
            if cap is not None:
                try:
                    item["exceeds_expected_cap"] = float(m.group(1)) > cap
                except ValueError:
                    pass
        else:
            item["found_value"] = None
        bench.append(item)

    total_w = sum(SEV_W[c["severity"]] for c in t["required_clauses"])
    got_w = sum(SEV_W[c["severity"]] * (1.0 if c["strength"] == "strong" else 0.8) for c in present)
    score = round(100 * got_w / total_w) if total_w else 0
    if score >= 85 and not flags:
        verdict = "MATCHES_TEMPLATE"
    elif score >= 60:
        verdict = "PARTIAL_MATCH"
    else:
        verdict = "TEMPLATE_NOT_MATCHING"

    return {
        "detected_type": tid, "detected_type_name": t["name"], "detection_confidence": conf,
        "detection_scores": scores, "candidate_types": sorted(scores, key=lambda k: -scores[k])[:3], "detection_note": ("Document type uncertain - confirm it or pass --type" if conf == "low" else ""), "match_score": score, "verdict": verdict,
        "missing_clauses": sorted(missing, key=lambda x: -SEV_W[x["severity"]]),
        "present_clauses": present,
        "red_flags": flags, "benchmarks": bench,
        "sources": [s["title"] for s in t["sources"]],
        "disclaimer": t["disclaimer"],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("file")
    ap.add_argument("--templates", default=DEFAULT_DIR)
    ap.add_argument("--type", default=None)
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args()
    text = read_text(a.file)
    if len(text.strip()) < 200:
        print("Very little text extracted (scanned PDF?). OCR needed.", file=sys.stderr)
    r = check_contract(text, a.templates, a.type)
    if a.json:
        print(json.dumps(r, indent=2))
        return
    print(f"Type: {r['detected_type_name']}  (confidence: {r['detection_confidence']})")
    if r["detection_note"]:
        print("NOTE:", r["detection_note"], "- candidates:", ", ".join(r["candidate_types"]))
    print(f"Match score: {r['match_score']}/100  ->  {r['verdict']}\n")
    if r["missing_clauses"]:
        print("MISSING clauses:")
        for m in r["missing_clauses"]:
            print(f"  [{m['severity']:6}] {m['title']}")
    thin = [m for m in r["present_clauses"] if m["strength"] != "strong"]
    if thin:
        print("\nFOUND but only one keyword matched (spot-check):")
        for m in thin:
            print(f"  [{m['severity']:6}] {m['title']}")
    if r["red_flags"]:
        print("\nRED FLAGS:")
        for f in r["red_flags"]:
            print(f"  - {f['title']}: \"{f['evidence'][:140]}\"")
    if r["benchmarks"]:
        print("\nBenchmarks:")
        for b in r["benchmarks"]:
            print(f"  - {b['label']}: found={b['found_value']}  expected {b['expected']}")


if __name__ == "__main__":
    main()
