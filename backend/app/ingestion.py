import re, fitz
from .models import Chunk
def _chunk(cid,n,start,end,section,clause,heading,text): return Chunk(chunk_id=f"{cid}_C{n:04d}",contract_id=cid,page_start=start,page_end=end,section=section,clause=clause,heading=heading,text=text,token_estimate=max(1,len(text)//4))
def extract_and_chunk(pdf,cid):
    doc=fitz.open(stream=pdf,filetype="pdf"); pages=[(i+1,p.get_text("text").strip()) for i,p in enumerate(doc)]; chunks=[]; current=[]; start=1; section=clause=heading=""
    for page,text in pages:
        if not text: continue
        for para in re.split(r"\n\s*\n",text):
            para=para.strip()
            if not para: continue
            m=re.match(r"^(\d+(?:\.\d+)*)[.)]?\s+(.+)$",para.replace("\n"," "))
            if m: clause=m.group(1); heading=m.group(2)[:140]; section=clause.split('.')[0]
            if current and sum(map(len,current))+len(para)>4200: chunks.append(_chunk(cid,len(chunks)+1,start,page,section,clause,heading,"\n\n".join(current))); current=[]; start=page
            current.append(para)
    if current: chunks.append(_chunk(cid,len(chunks)+1,start,pages[-1][0] if pages else 1,section,clause,heading,"\n\n".join(current)))
    return len(doc),chunks
