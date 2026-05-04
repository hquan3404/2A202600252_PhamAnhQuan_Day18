# Failure Analysis — Lab 18: Production RAG

**Nhóm:**   Phạm Anh Quân
**Thành viên:** [Phạm Anh Quân → M1, M2, M3, M4, M5]

---

## RAGAS Scores

| Metric | Naive Baseline | Production | Δ |
|--------|---------------|------------|---|
| Faithfulness | 0.0000 | 0.8000 | +0.8000 |
| Answer Relevancy | 0.0000 | 0.8408 | +0.8408 |
| Context Precision | 0.0000 | 0.8000 | +0.8000 |
| Context Recall | 0.0000 | 0.8000 | +0.8000 |

## Bottom-3 Failures

### #1
- **Question:** Bao lâu thì tôi phải thay đổi mật khẩu máy tính một lần?
- **Expected:** Mật khẩu máy tính bắt buộc phải thay đổi định kỳ 3 tháng 1 lần.
- **Got:** (Lỗi Hallucination - Trả về thông tin sai hoặc tự bịa)
- **Worst metric:** faithfulness (0.0000)
- **Error Tree:** Output sai → Context đúng? (Có) → Query OK? (Có)
- **Root cause:** LLM tự sáng tạo nội dung hoặc đưa thêm thông tin không có trong văn bản được truy xuất (LLM hallucinating).
- **Suggested fix:** Tighten prompt (ép chặt prompt "CHỈ TRẢ LỜI DỰA TRÊN CONTEXT"), lower temperature = 0.

### #2
- **Question:** Khi máy tính bị hỏng thì tôi cần gọi cho số nào để nhờ IT hỗ trợ?
- **Expected:** Bạn có thể liên hệ với bộ phận IT qua tổng đài nội bộ số 111.
- **Got:** (Lấy thừa context)
- **Worst metric:** context_precision (0.4999)
- **Error Tree:** Output OK → Context đúng? (Đúng một phần, lẫn rác) → Query OK? (Có)
- **Root cause:** Hệ thống tìm kiếm lấy về nhiều chunk không liên quan làm loãng context (Too many irrelevant chunks).
- **Suggested fix:** Tăng cường Reranking (CrossEncoder) hoặc thêm điều kiện Metadata filter theo chủ đề IT.

### #3
- **Question:** Nếu nghỉ ốm đột xuất thì tôi cần báo cáo như thế nào?
- **Expected:** Trong trường hợp nghỉ ốm đột xuất, nhân viên có thể báo cáo trực tiếp cho quản lý vào buổi sáng cùng ngày.
- **Got:** (Lấy thừa context)
- **Worst metric:** context_precision (0.4999)
- **Error Tree:** Output OK → Context đúng? (Đúng một phần, lẫn rác) → Query OK? (Có)
- **Root cause:** Hệ thống retrieval trả về nhiều văn bản dư thừa không chứa trực tiếp thông tin xin nghỉ ốm.
- **Suggested fix:** Áp dụng Semantic Chunking mạnh hơn hoặc cải tiến lại thuật toán Hybrid Search (điều chỉnh trọng số RRF).

## Case Study (cho presentation)

**Question chọn phân tích:** "Bao lâu thì tôi phải thay đổi mật khẩu máy tính một lần?"

**Error Tree walkthrough:**
1. Output đúng? → Không, trả về Faithfulness = 0.0 (LLM bị hallucination).
2. Context đúng? → Context Recall và Precision vẫn rất cao, tức là hệ thống đã tìm ra đoạn văn đúng ("Mật khẩu phải thay đổi mỗi 90 ngày").
3. Query rewrite OK? → Tốt.
4. Fix ở bước: Module Sinh câu trả lời (Generator / Pipeline).

**Nếu có thêm 1 giờ, sẽ optimize:**
- Viết lại hàm `run_query` trong `pipeline.py`: Cập nhật system prompt mạnh tay hơn: `Bạn là trợ lý công ty. NẾU VÀ CHỈ NẾU thông tin có trong context thì mới trả lời. Tuyệt đối không bịa đặt.`
- Hạ `temperature` của `ChatGoogleGenerativeAI` xuống `0.0`.
