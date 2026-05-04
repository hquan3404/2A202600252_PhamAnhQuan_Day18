"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation."""
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from datasets import Dataset
    import os
    
    from config import GEMINI_API_KEY
    if GEMINI_API_KEY:
        os.environ["GOOGLE_API_KEY"] = GEMINI_API_KEY
        
    dataset = Dataset.from_dict({
        "user_input": questions,
        "response": answers,
        "retrieved_contexts": contexts,
        "reference": ground_truths,
    })

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.embeddings import Embeddings
        from google import genai
        
        class CustomGeminiEmbeddings(Embeddings):
            def __init__(self, model_name="text-embedding-004"): # Trong genai SDK mới có thể dùng text-embedding-004
                self.client = genai.Client()
                self.model = model_name

            def embed_documents(self, texts: list[str]) -> list[list[float]]:
                # API của genai.Client()
                # embed_content cho danh sách văn bản
                result = self.client.models.embed_content(model=self.model, contents=texts)
                # result.embeddings là danh sách các object embedding, có thuộc tính values
                return [emb.values for emb in result.embeddings]

            def embed_query(self, text: str) -> list[float]:
                result = self.client.models.embed_content(model=self.model, contents=text)
                return result.embeddings[0].values

        chat_model = ChatGoogleGenerativeAI(model="gemini-3.1-flash-lite-preview", max_retries=5, timeout=60)
        embeddings_model = CustomGeminiEmbeddings(model_name="gemini-embedding-2") # Sử dụng model như user yêu cầu
        
        try:
            from ragas.llms import LangchainLLMWrapper
            from ragas.embeddings import LangchainEmbeddingsWrapper
            eval_llm = LangchainLLMWrapper(chat_model)
            eval_embeddings = LangchainEmbeddingsWrapper(embeddings_model)
            result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall], llm=eval_llm, embeddings=eval_embeddings)
        except ImportError:
            result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall], llm=chat_model, embeddings=embeddings_model)
    except Exception as e:
        print(f"Warning: Gemini config failed, using default Ragas config. Error: {e}")
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy, context_precision, context_recall])

    df = result.to_pandas()
    per_question = []
    for _, row in df.iterrows():
        per_question.append(EvalResult(
            question=row.get("user_input", row.get("question", "")),
            answer=row.get("response", row.get("answer", "")),
            contexts=row.get("retrieved_contexts", row.get("contexts", [])),
            ground_truth=row.get("reference", row.get("ground_truth", "")),
            faithfulness=float(row.get("faithfulness", 0.0) or 0.0),
            answer_relevancy=float(row.get("answer_relevancy", 0.0) or 0.0),
            context_precision=float(row.get("context_precision", 0.0) or 0.0),
            context_recall=float(row.get("context_recall", 0.0) or 0.0)
        ))

    return {
        "faithfulness": float(df["faithfulness"].mean()) if "faithfulness" in df else 0.0,
        "answer_relevancy": float(df["answer_relevancy"].mean()) if "answer_relevancy" in df else 0.0,
        "context_precision": float(df["context_precision"].mean()) if "context_precision" in df else 0.0,
        "context_recall": float(df["context_recall"].mean()) if "context_recall" in df else 0.0,
        "per_question": per_question
    }


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    analysis = []
    for res in eval_results:
        scores = {
            "faithfulness": res.faithfulness,
            "answer_relevancy": res.answer_relevancy,
            "context_precision": res.context_precision,
            "context_recall": res.context_recall
        }
        valid_scores = [v for v in scores.values() if v == v and v is not None]
        avg_score = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
        
        worst_metric = min(scores, key=lambda k: scores[k] if scores[k] == scores[k] else float('inf'))
        worst_score = scores[worst_metric]
        
        diagnosis, fix = "Pipeline OK", "No fix needed"
        if worst_metric == "faithfulness" and worst_score < 0.85:
            diagnosis, fix = "LLM hallucinating", "Tighten prompt, lower temperature"
        elif worst_metric == "context_recall" and worst_score < 0.75:
            diagnosis, fix = "Missing relevant chunks", "Improve chunking or add BM25"
        elif worst_metric == "context_precision" and worst_score < 0.75:
            diagnosis, fix = "Too many irrelevant chunks", "Add reranking or metadata filter"
        elif worst_metric == "answer_relevancy" and worst_score < 0.80:
            diagnosis, fix = "Answer doesn't match question", "Improve prompt template"
            
        analysis.append({
            "question": res.question,
            "worst_metric": worst_metric,
            "score": worst_score,
            "avg_score": avg_score,
            "diagnosis": diagnosis,
            "suggested_fix": fix
        })
        
    analysis.sort(key=lambda x: x["avg_score"])
    return analysis[:bottom_n]


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
