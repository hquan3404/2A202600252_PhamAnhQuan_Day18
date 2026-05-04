"""
Module 5: Enrichment Pipeline
==============================
Làm giàu chunks TRƯỚC khi embed: Summarize, HyQA, Contextual Prepend, Auto Metadata.

Test: pytest tests/test_m5.py
"""

import os, sys
from dataclasses import dataclass, field

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import GEMINI_API_KEY
import google.generativeai as genai

def get_gemini_model():
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        return genai.GenerativeModel("gemini-3.1-flash-lite-preview")
    return None


@dataclass
class EnrichedChunk:
    """Chunk đã được làm giàu."""
    original_text: str
    enriched_text: str
    summary: str
    hypothesis_questions: list[str]
    auto_metadata: dict
    method: str  # "contextual", "summary", "hyqa", "full"


# ─── Technique 1: Chunk Summarization ────────────────────


def summarize_chunk(text: str) -> str:
    """
    Tạo summary ngắn cho chunk.
    Embed summary thay vì (hoặc cùng với) raw chunk → giảm noise.

    Args:
        text: Raw chunk text.

    Returns:
        Summary string (2-3 câu).
    """
    model = get_gemini_model()
    if not model:
        sentences = text.split(". ")
        return ". ".join(sentences[:2]) + ("." if sentences else "")
        
    try:
        resp = model.generate_content(f"Tóm tắt đoạn văn sau trong 2-3 câu ngắn gọn bằng tiếng Việt:\n\n{text}")
        return resp.text.strip()
    except Exception as e:
        sentences = text.split(". ")
        return ". ".join(sentences[:2]) + ("." if sentences else "")


# ─── Technique 2: Hypothesis Question-Answer (HyQA) ─────


def generate_hypothesis_questions(text: str, n_questions: int = 3) -> list[str]:
    """
    Generate câu hỏi mà chunk có thể trả lời.
    Index cả questions lẫn chunk → query match tốt hơn (bridge vocabulary gap).

    Args:
        text: Raw chunk text.
        n_questions: Số câu hỏi cần generate.

    Returns:
        List of question strings.
    """
    model = get_gemini_model()
    if not model: return []
    try:
        resp = model.generate_content(f"Dựa trên đoạn văn, tạo {n_questions} câu hỏi mà đoạn văn có thể trả lời. Trả về mỗi câu hỏi trên 1 dòng.\n\n{text}")
        questions = resp.text.strip().split("\n")
        return [q.strip().lstrip("0123456789.-* ") for q in questions if q.strip()]
    except Exception:
        return []


# ─── Technique 3: Contextual Prepend (Anthropic style) ──


def contextual_prepend(text: str, document_title: str = "") -> str:
    """
    Prepend context giải thích chunk nằm ở đâu trong document.
    Anthropic benchmark: giảm 49% retrieval failure (alone).

    Args:
        text: Raw chunk text.
        document_title: Tên document gốc.

    Returns:
        Text với context prepended.
    """
    model = get_gemini_model()
    if not model: return text
    try:
        prompt = f"Viết 1 câu ngắn mô tả đoạn văn này nằm ở đâu trong tài liệu và nói về chủ đề gì. Chỉ trả về 1 câu.\n\nTài liệu: {document_title}\n\nĐoạn văn:\n{text}"
        resp = model.generate_content(prompt)
        context = resp.text.strip()
        return f"{context}\n\n{text}"
    except Exception:
        return text


# ─── Technique 4: Auto Metadata Extraction ──────────────


def extract_metadata(text: str) -> dict:
    """
    LLM extract metadata tự động: topic, entities, date_range, category.

    Args:
        text: Raw chunk text.

    Returns:
        Dict with extracted metadata fields.
    """
    model = get_gemini_model()
    if not model: return {}
    try:
        prompt = f'Trích xuất metadata từ đoạn văn. Chỉ trả về một khối JSON hợp lệ duy nhất, KHÔNG kèm giải thích thêm. Định dạng mẫu: {{"topic": "...", "entities": ["..."], "category": "policy|hr|it|finance", "language": "vi|en"}}\n\nĐoạn văn:\n{text}'
        resp = model.generate_content(prompt)
        import json, re
        match = re.search(r'\{.*\}', resp.text, re.DOTALL)
        if match:
            return json.loads(match.group(0))
        return {}
    except Exception:
        return {}


# ─── Full Enrichment Pipeline ────────────────────────────


def enrich_chunks(
    chunks: list[dict],
    methods: list[str] | None = None,
) -> list[EnrichedChunk]:
    """
    Chạy enrichment pipeline trên danh sách chunks.

    Args:
        chunks: List of {"text": str, "metadata": dict}
        methods: List of methods to apply. Default: ["contextual", "hyqa", "metadata"]
                 Options: "summary", "hyqa", "contextual", "metadata", "full"

    Returns:
        List of EnrichedChunk objects.
    """
    if methods is None:
        methods = ["contextual", "hyqa", "metadata"]

    enriched = []

    for chunk in chunks:
        t = chunk["text"]
        metadata = chunk.get("metadata", {})
        
        summary = summarize_chunk(t) if "summary" in methods or "full" in methods else ""
        questions = generate_hypothesis_questions(t) if "hyqa" in methods or "full" in methods else []
        enriched_text = contextual_prepend(t, metadata.get("source", "")) if "contextual" in methods or "full" in methods else t
        auto_meta = extract_metadata(t) if "metadata" in methods or "full" in methods else {}
        
        final_meta = metadata.copy()
        final_meta.update(auto_meta)
        
        enriched.append(EnrichedChunk(
            original_text=t,
            enriched_text=enriched_text,
            summary=summary,
            hypothesis_questions=questions,
            auto_metadata=final_meta,
            method="+".join(methods)
        ))
        
    return enriched


# ─── Main ────────────────────────────────────────────────

if __name__ == "__main__":
    sample = "Nhân viên chính thức được nghỉ phép năm 12 ngày làm việc mỗi năm. Số ngày nghỉ phép tăng thêm 1 ngày cho mỗi 5 năm thâm niên công tác."

    print("=== Enrichment Pipeline Demo ===\n")
    print(f"Original: {sample}\n")

    s = summarize_chunk(sample)
    print(f"Summary: {s}\n")

    qs = generate_hypothesis_questions(sample)
    print(f"HyQA questions: {qs}\n")

    ctx = contextual_prepend(sample, "Sổ tay nhân viên VinUni 2024")
    print(f"Contextual: {ctx}\n")

    meta = extract_metadata(sample)
    print(f"Auto metadata: {meta}")
