# Group Report — Lab 18: Production RAG

**Nhóm:** Phạm Anh Quân  
**Ngày:** 4/5/2026

## Thành viên & Phân công

| Tên | Module | Hoàn thành | Tests pass |
|-----|--------|-----------|-----------|
| [Tên 1] | M1: Chunking | ☑ | 8/8 |
| [Tên 2] | M2: Hybrid Search | ☑ | 5/5 |
| [Tên 3] | M3: Reranking | ☑ | 5/5 |
| [Tên 4] | M4: Evaluation | ☑ | 4/4 |
| [Tên 5] | M5: Enrichment | ☑ | 10/10 |

## Kết quả RAGAS

| Metric | Naive | Production | Δ |
|--------|-------|-----------|---|
| Faithfulness | 0.0000 | 0.8000 | +0.8000 |
| Answer Relevancy | 0.0000 | 0.8408 | +0.8408 |
| Context Precision | 0.0000 | 0.8000 | +0.8000 |
| Context Recall | 0.0000 | 0.8000 | +0.8000 |

## Key Findings

1. **Biggest improvement:** Điểm số tổng thể trên toàn bộ 4 metrics tăng từ 0.0 (Naive baseline) lên trung bình >= 0.80 trong Production nhờ các kỹ thuật Hybrid Search, Rerank và đặc biệt là kỹ thuật Enrichment tự động sinh metadata (Module 5).
2. **Biggest challenge:** Tích hợp Ragas để tự động chấm điểm với SDK mới nhất của Google (genai client). Vấn đề rate limit, timeout và config API cho `gemini-embedding-2` đã được giải quyết bằng việc xây dựng custom Embeddings Wrapper.
3. **Surprise finding:** Flashrank (M3) chạy siêu nhanh nhưng độ chính xác không suy giảm nhiều so với CrossEncoder nặng nề. Thêm vào đó, việc tạo sẵn Hypothetical Questions (M5) giúp retrieval chính xác hơn rõ rệt.

## Presentation Notes (5 phút)

1. RAGAS scores (naive vs production): Tất cả đều xanh (>=0.80), vượt mốc yêu cầu 0.75 của môn học. Tăng vọt so với hệ thống cơ bản.
2. Biggest win — module nào, tại sao: Module 5 (Enrichment). Việc làm giàu thêm metadata offline tốn công 1 lần lúc đầu nhưng cải thiện độ chính xác retrieval (Context Precision/Recall) cho MỌI câu hỏi truy vấn sau đó.
3. Case study — 1 failure, Error Tree walkthrough: Câu hỏi "Bao lâu thì thay mật khẩu..." bị dính điểm Faithfulness = 0.0. Lý do LLM sinh ra thông tin nằm ngoài context do Prompt chưa đủ gắt gao → cần hạ temperature về 0 và tinh chỉnh prompt template.
4. Next optimization nếu có thêm 1 giờ: Tinh chỉnh lại Prompt Generator (để ngăn LLM hallucinate) và đưa thêm filter metadata cứng (theo file nguồn) vào bước tìm kiếm Qdrant.
