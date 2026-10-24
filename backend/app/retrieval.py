import re
import numpy as np
from rank_bm25 import BM25Okapi
from .config import settings
from .vector_store import search as vector_search


class HybridRetriever:
    _cache = {}
    STOPWORDS = {
        "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
        "how", "in", "is", "it", "of", "on", "or", "that", "the", "this",
        "to", "what", "when", "where", "who", "with",
    }

    def __init__(self, chunks, vectors=None, root=None, embed_query=None, contract_id=None, retrieval_config=None):
        self.chunks = chunks
        self.root = root
        self.tokens = [re.findall(r"\w+", chunk.text.lower()) for chunk in chunks]
        self.contract_id = contract_id or (chunks[0].contract_id if chunks else "")
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        cached = self._cache.get(self.contract_id)
        if cached and cached[0] == chunk_ids:
            self.bm25 = cached[1]
        else:
            self.bm25 = BM25Okapi(self.tokens) if chunks else None
            if self.contract_id:
                self._cache[self.contract_id] = (chunk_ids, self.bm25)
        self.vectors = vectors
        self.embed_query = embed_query
        self.retrieval_config = retrieval_config or {}

    def search(self, query, top_k=8):
        q = {word for word in re.findall(r"\w+", query.lower()) if word not in self.STOPWORDS}
        lexical = []
        if self.bm25 and q:
            lexical_scores = self.bm25.get_scores(list(q))
            lexical_k = int(self.retrieval_config.get("top_k_bm25", settings.top_k_bm25))
            lexical = [
                int(index)
                for index in np.argsort(lexical_scores)[::-1][:lexical_k]
                if lexical_scores[index] > 0
            ]
        semantic = []
        if self.embed_query and self.contract_id:
            ids = vector_search(
                self.contract_id,
                self.embed_query(query),
                min(int(self.retrieval_config.get("top_k_vector", settings.top_k_vector)), len(self.chunks)),
            )
            positions = {chunk.chunk_id: index for index, chunk in enumerate(self.chunks)}
            semantic = [positions[chunk_id] for chunk_id in ids if chunk_id in positions]
        scores = {}
        for rank, index in enumerate(lexical):
            scores[index] = scores.get(index, 0) + 1 / (60 + rank)
        for rank, index in enumerate(semantic):
            scores[index] = scores.get(index, 0) + 1 / (60 + rank)
        for index, chunk in enumerate(self.chunks):
            text = chunk.text.lower()
            exact = sum(1 for token in q if len(token) > 3 and re.search(rf"\b{re.escape(token)}\b", text))
            if chunk.clause and re.search(rf"\b{re.escape(chunk.clause.lower())}\b", query.lower()):
                scores[index] = scores.get(index, 0) + 0.12
            if exact:
                scores[index] = scores.get(index, 0) + min(exact, 8) * 0.005
        order = sorted(scores, key=scores.get, reverse=True)
        return [self.chunks[index] for index in order[:top_k]] or self._fallback(q, top_k)

    def _fallback(self, q, top_k):
        scored = []
        for chunk in self.chunks:
            words = re.findall(r"\w+", chunk.text.lower())
            scored.append((len(q & set(words)) / (len(q) + 1), chunk))
        return [chunk for _, chunk in sorted(scored, key=lambda item: item[0], reverse=True)[:top_k]]
