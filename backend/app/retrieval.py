import re, math, json
from collections import Counter
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from .config import settings
class HybridRetriever:
    def __init__(self,chunks,vectors=None,root=None,embed_query=None):
        self.chunks=chunks; self.root=root
        self.tokens=[re.findall(r"\w+",c.text.lower()) for c in chunks]
        self.bm25=BM25Okapi(self.tokens) if chunks else None
        if root and chunks: (root/"indexes"/f"{chunks[0].contract_id}.bm25.json").write_text(json.dumps(self.tokens),encoding="utf-8")
        self.vectors=np.asarray(vectors,dtype="float32") if vectors is not None else None; self.embed_query=embed_query
        self.index=None
        if self.vectors is not None and len(self.vectors):
            faiss.normalize_L2(self.vectors); self.index=faiss.IndexFlatIP(self.vectors.shape[1]); self.index.add(self.vectors)
            if root: faiss.write_index(self.index,str(root/"indexes"/f"{chunks[0].contract_id}.faiss"))
    def search(self,query,top_k=8):
        q=set(re.findall(r"\w+",query.lower())); lexical=[]
        if self.bm25:
            lexical=[int(x) for x in np.argsort(self.bm25.get_scores(list(q)))[::-1][:settings.top_k_bm25]]
        semantic=[]
        if self.index is not None and self.embed_query:
            v=np.asarray([self.embed_query(query)],dtype="float32"); faiss.normalize_L2(v); _,idx=self.index.search(v,min(settings.top_k_vector,len(self.chunks))); semantic=[int(x) for x in idx[0] if x>=0]
        order=[]
        for rank,idx in enumerate(lexical+semantic):
            if idx not in order: order.append(idx)
        return [self.chunks[i] for i in order[:top_k]] or self._fallback(q,top_k)
    def _fallback(self,q,top_k):
        scored=[]
        for c in self.chunks:
            words=re.findall(r"\w+",c.text.lower()); scored.append((len(q&set(words))/(len(q)+1),c))
        return [c for _,c in sorted(scored,key=lambda x:x[0],reverse=True)[:top_k]]
