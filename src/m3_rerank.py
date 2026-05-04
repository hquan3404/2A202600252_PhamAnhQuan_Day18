"""Module 3: Reranking — Cross-encoder top-20 → top-3 + latency benchmark."""

import os, sys, time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None

    def _load_model(self):
        if self._model is None:
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self.model_name)
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents: top-20 → top-k."""
        if not documents:
            return []
        model = self._load_model()
        pairs = [(query, doc["text"]) for doc in documents]
        scores = model.predict(pairs)
        
        combined = [(float(score), doc) for score, doc in zip(scores, documents)]
        combined.sort(key=lambda x: x[0], reverse=True)
        
        results = []
        for i, (score, doc) in enumerate(combined[:top_k]):
            results.append(RerankResult(
                text=doc["text"],
                original_score=doc.get("score", 0.0),
                rerank_score=score,
                metadata=doc.get("metadata", {}),
                rank=i+1
            ))
        return results


class FlashrankReranker:
    """Lightweight alternative (<5ms). Optional."""
    def __init__(self):
        self._model = None

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents: return []
        if self._model is None:
            from flashrank import Ranker
            self._model = Ranker()
        from flashrank import RerankRequest
        
        passages = [{"id": i, "text": d["text"], "meta": d.get("metadata", {})} for i, d in enumerate(documents)]
        req = RerankRequest(query=query, passages=passages)
        results = self._model.rerank(req)
        
        final_results = []
        for i, r in enumerate(results[:top_k]):
            idx = r.get("id", 0)
            orig_doc = documents[idx] if isinstance(idx, int) and idx < len(documents) else documents[0]
            final_results.append(RerankResult(
                text=r["text"],
                original_score=orig_doc.get("score", 0.0),
                rerank_score=r.get("score", 0.0),
                metadata=r.get("meta", {}),
                rank=i+1
            ))
        return final_results


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs."""
    times = []
    if documents:
        reranker.rerank(query, documents) # Warmup
        
    for _ in range(n_runs):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        times.append((time.perf_counter() - start) * 1000)
        
    if not times:
        return {"avg_ms": 0, "min_ms": 0, "max_ms": 0}
        
    return {"avg_ms": sum(times)/len(times), "min_ms": min(times), "max_ms": max(times)}


if __name__ == "__main__":
    query = "Nhân viên được nghỉ phép bao nhiêu ngày?"
    docs = [
        {"text": "Nhân viên được nghỉ 12 ngày/năm.", "score": 0.8, "metadata": {}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "score": 0.7, "metadata": {}},
        {"text": "Thời gian thử việc là 60 ngày.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
