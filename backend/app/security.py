import re
SUSPICIOUS=re.compile(r"ignore\s+(previous|prior)|system\s+prompt|developer\s+message|api\s*key|reveal\s+secret|access\s+another\s+contract",re.I)
def classify_untrusted(text): return {"suspicious":bool(SUSPICIOUS.search(text)),"matches":SUSPICIOUS.findall(text)[:10]}
def valid_evidence(finding,chunks):
    by_id={c.chunk_id:c for c in chunks}
    return bool(finding.evidence) and all((c:=by_id.get(e.chunk_id)) and e.page>=c.page_start and e.page<=c.page_end and e.quote in c.text for e in finding.evidence)
