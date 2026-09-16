"""Compute actual retrieval measurements against labeled synthetic fixtures."""
from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).parents[1]))
from backend.app.models import Chunk
from backend.app.retrieval import HybridRetriever
def recall_at_k(results,expected,k): return int(any(x.chunk_id in expected for x in results[:k]))
if __name__=="__main__":
    text=Path(__file__).with_name("synthetic_contract.txt").read_text(); chunks=[Chunk(chunk_id=f"c{i}",contract_id="synthetic",page_start=1,page_end=1,text=x) for i,x in enumerate(text.splitlines()) if x.strip()]; r=HybridRetriever(chunks).search("unlimited liability",5); print({"recall@5":recall_at_k(r,{"c1"},5),"returned":len(r)})
