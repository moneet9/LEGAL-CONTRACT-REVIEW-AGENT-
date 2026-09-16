from datetime import datetime, timezone
from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from .config import settings
from .models import ContractMeta, ReviewRequest, DecisionRequest, ChatRequest, WorkspaceChatRequest
from .storage import store
from .ingestion import extract_and_chunk
from .orchestrator import MasterOrchestratorAgent
from .security import classify_untrusted
from .gemini import GeminiClient
from .config import settings
import numpy as np
app=FastAPI(title="PS-9 Legal Contract Review Agent",version="1.0.0")
app.add_middleware(CORSMiddleware,allow_origins=["http://localhost:5173"],allow_methods=["*"],allow_headers=["*"])
gemini=GeminiClient()
def scoped_meta(cid,workspace_id):
    meta=store.get_meta(cid)
    if meta.workspace_id != workspace_id: raise HTTPException(404,"Contract not found in this workspace")
    return meta
@app.get("/api/health")
def health(): return {"status":"ok","gemini_configured":bool(settings.gemini_api_key)}
@app.post("/api/contracts",response_model=ContractMeta)
async def upload(file:UploadFile=File(...),x_workspace_id: str = Header(default="default")):
    content=await file.read()
    if file.content_type!="application/pdf" or not content.startswith(b"%PDF"): raise HTTPException(400,"Only valid PDF files are accepted")
    cid,digest=store.contract_id(content,x_workspace_id); pages,chunks=extract_and_chunk(content,cid)
    vectors=gemini.embed([c.text for c in chunks]) if gemini.available and chunks else None
    embedding_status="gemini_saved" if vectors else "fallback_no_gemini_key"
    if vectors: store.save_embeddings(cid,vectors)
    meta=ContractMeta(contract_id=cid,original_filename=file.filename or "contract.pdf",sha256=digest,upload_time=datetime.now(timezone.utc),workspace_id=x_workspace_id,page_count=pages,processing_status="ready",chunk_count=len(chunks),embedding_status=embedding_status)
    store.save_contract(meta,content); store.save_chunks(cid,chunks); store.audit({"contract_id":cid,"workspace_id":x_workspace_id,"agent":"Ingestion","action":"indexed","page_count":pages,"chunk_count":len(chunks),"embedding_status":embedding_status,"suspicious_content":classify_untrusted(" ".join(c.text for c in chunks))["suspicious"]})
    return meta
@app.get("/api/contracts",response_model=list[ContractMeta])
def contracts(x_workspace_id: str = Header(default="default")):
    return store.all_meta(x_workspace_id)
@app.get("/api/contracts/{cid}")
def get_contract(cid:str,x_workspace_id: str = Header(default="default")):
    try: return scoped_meta(cid,x_workspace_id)
    except KeyError: raise HTTPException(404,"Contract not found")
@app.get("/api/contracts/{cid}/document")
def document(cid:str,x_workspace_id: str = Header(default="default")):
    try: meta=scoped_meta(cid,x_workspace_id)
    except KeyError: raise HTTPException(404,"Contract not found")
    return FileResponse(settings.data_dir/"contracts"/f"{cid}.pdf",media_type="application/pdf",filename=meta.original_filename)
@app.get("/api/contracts/{cid}/messages")
def messages(cid:str,x_workspace_id: str = Header(default="default")):
    try: scoped_meta(cid,x_workspace_id)
    except KeyError: raise HTTPException(404,"Contract not found")
    return [m.model_dump() for m in store.messages(cid)]
@app.get("/api/contracts/{cid}/activity")
def activity(cid:str,x_workspace_id: str = Header(default="default")):
    try: meta=scoped_meta(cid,x_workspace_id)
    except KeyError: raise HTTPException(404,"Contract not found")
    path=settings.data_dir/"audit"/"events.jsonl"
    if not path.exists(): return []
    import json
    return [x for x in (json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line) if x.get("contract_id")==cid and x.get("workspace_id",meta.workspace_id)==x_workspace_id]
@app.post("/api/contracts/{cid}/review")
def review(cid:str,req:ReviewRequest,x_workspace_id: str = Header(default="default")):
    try: chunks=store.load_chunks(cid); meta=scoped_meta(cid,x_workspace_id)
    except (KeyError,FileNotFoundError): raise HTTPException(404,"Contract not found or not indexed")
    vectors=None
    try: vectors=store.load_embeddings(cid)
    except FileNotFoundError: pass
    embed_query=(lambda q: gemini.embed([q])[0]) if gemini.available else None
    agent=MasterOrchestratorAgent(chunks,lambda e:store.audit({"contract_id":cid,**e}),vectors,embed_query,settings.data_dir,gemini); findings=agent.run(req.question); store.save_findings(cid,findings); meta.analysis_status="complete"; store.update_meta(meta)
    return {"contract_id":cid,"question":req.question,"findings":[f.model_dump() for f in findings],"trace":["Master Orchestrator","Planner","Selected Specialist Agents","Cross-Clause Agent","Evidence Verification","Judge","Human Review Gate"]}
@app.post("/api/contracts/{cid}/chat")
def chat(cid:str,req:ChatRequest,x_workspace_id: str = Header(default="default")):
    try: chunks=store.load_chunks(cid); scoped_meta(cid,x_workspace_id)
    except (KeyError,FileNotFoundError): raise HTTPException(404,"Contract not found or not indexed")
    store.save_message(cid,"user",req.message)
    vectors=None
    try: vectors=store.load_embeddings(cid)
    except FileNotFoundError: pass
    from .retrieval import HybridRetriever
    embed_query=(lambda q: gemini.embed([q])[0]) if gemini.available else None
    hits=HybridRetriever(chunks,vectors,settings.data_dir,embed_query).search(req.message,top_k=4)
    evidence=[{"page":h.page_start,"clause":h.clause,"quote":h.text[:380],"chunk_id":h.chunk_id} for h in hits]
    result=gemini.generate_structured("Answer using only this contract evidence. If insufficient, say so. Question: "+req.message+"\nEvidence:\n"+"\n---\n".join(h.text for h in hits), {"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}) if gemini.available else None
    text=result["answer"] if result else ("Based on the indexed contract: " + (hits[0].text[:700] if hits else "I could not find supporting language in this document."))
    return store.save_message(cid,"assistant",text,evidence).model_dump()
@app.get("/api/workspace/messages")
def workspace_messages(x_workspace_id: str = Header(default="default")):
    return [m.model_dump() for m in store.workspace_messages(x_workspace_id)]
@app.post("/api/workspace/chat")
def workspace_chat(req:WorkspaceChatRequest,x_workspace_id: str = Header(default="default")):
    ids=list(dict.fromkeys(req.contract_ids))
    if not ids: raise HTTPException(400,"Select at least one contract")
    selected=[]
    for cid in ids:
        try: selected.append((cid,scoped_meta(cid,x_workspace_id),store.load_chunks(cid)))
        except (KeyError,FileNotFoundError): raise HTTPException(404,f"Contract not found or not indexed: {cid}")
    store.save_workspace_message("user",req.message,workspace_id=x_workspace_id)
    from .retrieval import HybridRetriever
    all_hits=[]
    for cid,meta,chunks in selected:
        vectors=None
        try: vectors=store.load_embeddings(cid)
        except FileNotFoundError: pass
        embed_query=(lambda q: gemini.embed([q])[0]) if gemini.available else None
        for hit in HybridRetriever(chunks,vectors,settings.data_dir,embed_query).search(req.message,top_k=3):
            all_hits.append((meta.original_filename,hit))
    evidence=[{"page":h.page_start,"clause":h.clause,"quote":f"[{name}] {h.text[:360]}","chunk_id":h.chunk_id} for name,h in all_hits]
    context="\n---\n".join(f"DOCUMENT: {name}\n{h.text}" for name,h in all_hits)
    result=gemini.generate_structured("Answer using only the supplied contract evidence. Compare documents when useful. If insufficient, say so. Question: "+req.message+"\nEvidence:\n"+context, {"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"]}) if gemini.available else None
    text=result["answer"] if result else ("Across the selected contracts: " + (all_hits[0][1].text[:700] if all_hits else "I could not find supporting language in the selected documents."))
    return store.save_workspace_message("assistant",text,evidence,workspace_id=x_workspace_id).model_dump()
@app.get("/api/contracts/{cid}/review")
def get_review(cid:str,x_workspace_id: str = Header(default="default")):
    try: scoped_meta(cid,x_workspace_id)
    except KeyError: raise HTTPException(404,"Contract not found")
    return {"contract_id":cid,"findings":[f.model_dump()|{"decision":d} for f,d in store.findings(cid)]}
@app.post("/api/findings/{fid}/decision")
def decision(fid:str,req:DecisionRequest):
    if req.decision not in {"approved","rejected","more_evidence","edited"}: raise HTTPException(400,"Invalid decision")
    store.decide(fid,req.decision); store.audit({"agent":"HumanReview","action":"decision","finding_id":fid,"decision":req.decision,"note":req.note}); return {"status":"recorded"}
