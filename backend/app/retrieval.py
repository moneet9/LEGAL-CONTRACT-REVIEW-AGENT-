import re, json
import numpy as np
from rank_bm25 import BM25Okapi
from .config import settings
from .vector_store import search as vector_search


class HybridRetriever:
    def __init__(self,chunks,vectors=None,root=None,embed_query=None,contract_id=None):
        self.chunks=chunks; self.root=root
        self.tokens=[re.findall(r"\w+",c.text.lower()) for c in chunks]
        self.bm25=BM25Okapi(self.tokens) if chunks else None
        if root and chunks: (root/"indexes"/f"{chunks[0].contract_id}.bm25.json").write_text(json.dumps(self.tokens),encoding="utf-8")
        self.vectors=vectors; self.embed_query=embed_query; self.contract_id=contract_id or (chunks[0].contract_id if chunks else "")
    def search(self,query,top_k=8):
        q=set(re.findall(r"\w+",query.lower())); lexical=[]
        if self.bm25:
            lexical=[int(x) for x in np.argsort(self.bm25.get_scores(list(q)))[::-1][:settings.top_k_bm25]]
        semantic=[]
        if self.embed_query and self.contract_id:
            ids=vector_search(self.contract_id,self.embed_query(query),min(settings.top_k_vector,len(self.chunks)))
            positions={chunk.chunk_id:index for index,chunk in enumerate(self.chunks)}
            semantic=[positions[chunk_id] for chunk_id in ids if chunk_id in positions]
        scores={}
        for rank,index in enumerate(lexical): scores[index]=scores.get(index,0)+1/(60+rank)
        for rank,index in enumerate(semantic): scores[index]=scores.get(index,0)+1/(60+rank)
        for index,chunk in enumerate(self.chunks):
            text=chunk.text.lower(); exact=sum(1 for token in q if len(token)>3 and token in text)
            if chunk.clause and chunk.clause.lower() in query.lower(): scores[index]=scores.get(index,0)+0.12
            scores[index]=scores.get(index,0)+min(exact,8)*0.005
        order=sorted(scores,key=scores.get,reverse=True)
        return [self.chunks[index] for index in order[:top_k]] or self._fallback(q,top_k)
    def _fallback(self,q,top_k):
        scored=[]
        for c in self.chunks:
            words=re.findall(r"\w+",c.text.lower()); scored.append((len(q&set(words))/(len(q)+1),c))
        return [c for _,c in sorted(scored,key=lambda x:x[0],reverse=True)[:top_k]]
