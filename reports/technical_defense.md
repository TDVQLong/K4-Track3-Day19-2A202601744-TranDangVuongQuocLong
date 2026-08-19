# 🛡️ Báo Cáo Thuyết Minh Kỹ Thuật (Technical Defense)

**Họ và tên học viên:** Trần Đặng Vương Quốc Long  
**Mã học viên:** 2A202601744  
**Môn học:** Track 3 — Production-Grade GraphRAG vs Flat RAG  

---

## 1. Coreference Resolution & Entity Extraction
- **Tình huống phân giải sai:** Khi xử lý các văn bản tin tức công nghệ dài có nhiều thực thể cùng giới tính/đại từ (ví dụ: *"Satya Nadella met Sam Altman... He announced..."*), mô hình LLM Coreference Resolution có thể nhầm lẫn `He` thành `Satya Nadella` thay vì `Sam Altman`.
- **Hậu quả đối với Knowledge Graph:** Dẫn đến việc gán nhầm hành động/sự kiện cho sai thực thể, sinh ra **False Edges** (ví dụ: `Satya Nadella -ANNOUNCED-> Product_X` thay vì `Sam Altman`), gây sai lệch tri thức và làm suy giảm Faithfulness score khi RAG truy vấn.

---

## 2. Entity Resolution Threshold & Lexical Guard
- **Ngưỡng Cosine Similarity:** Chọn ngưỡng `0.85` cho Vector ANN Candidate Search.
- **Cặp thực thể bị Lexical Guard chặn:** 
  - `Apple Inc.` (Company) và `Apple Music` (Product/Service).
  - **Lý do:** Mặc dù embedding vector của hai thực thể này rất gần nhau ($> 0.88$) do xuất hiện trong ngữ cảnh công nghệ tương tự, Lexical Guard chặn gộp dựa trên quy tắc khác biệt `entity_type` (Company vs Product) và tỉ lệ tương đồng từ vựng (Jaro-Winkler / Levenshtein ratio) nhằm tránh làm sụp đổ các nút khái niệm khác biệt vào làm một.

---

## 3. Super-node Analysis
- **Top 3 thực thể có bậc (degree) cao nhất:**
  1. `Microsoft` (Company) — Bậc $> 120$
  2. `OpenAI` (Company) — Bậc $> 105$
  3. `Google` (Company) — Bậc $> 98$
- **Đánh giá chính sách ưu tiên 50 cạnh mới nhất (`published_date DESC LIMIT 50`):**
  - *Ưu điểm:* Kiểm soát được bùng nổ ngữ cảnh (Context Explosion), giữ số lượng token nạp vào LLM dưới ngưỡng `MAX_GRAPH_CONTEXT_CHARS = 14000`, ưu tiên tin tức cập nhật mới nhất.
  - *Rủi ro:* Có thể bỏ sót các mối quan hệ quan trọng xảy ra trong quá khứ xa hơn nếu câu hỏi của người dùng truy vấn lịch sử giai đoạn cũ.

---

## 4. Benchmark Quality vs Latency vs Token Usage
- **Bảng so sánh tổng hợp:**
  - **Flat RAG:** Latency trung bình ~12.3s | Token usage ~2,100 tokens/query | Comprehensiveness: 3.4/5 | Faithfulness: 4.2/5 | Multi-hop: 2.2/5
  - **GraphRAG (Hybrid):** Latency trung bình ~27.1s | Token usage ~4,800 tokens/query | Comprehensiveness: 4.8/5 | Faithfulness: 4.6/5 | Multi-hop: 4.8/5
- **Nhận xét:** GraphRAG tốn Latency gấp ~2.2 lần và Token gấp ~2.3 lần nhưng mang lại hiệu quả vượt trội ở các câu hỏi suy luận phức tạp (Multi-hop Reasoning).

---

## 5. Trade-offs, Agent Control & Scale 350MB
- **Quyết định từ chối AI Coding Agent:** AI Agent đề xuất dùng Pairwise Cosine Similarity $O(N^2)$ cho toàn bộ thực thể. Tôi đã từ chối vì độ phức tạp $O(N^2)$ gây bùng nổ bộ nhớ (OOM) và thời gian chạy tăng theo cấp số nhân. Giải pháp thay thế: Dùng FAISS Index (ANN) kết hợp Blocking theo `entity_type` và Lexical Guard.
- **Giải pháp Scale 350MB (~100,000 bài báo):**
  1. *Extraction Latency:* Dùng Async Queue Worker (Celery/RabbitMQ) kết hợp mô hình SLM chuyên dụng local (GLiNER/NuExtract).
  2. *Graph Traversal:* Dùng thuật toán **Leiden / Louvain Community Detection** từ Neo4j GDS để tạo Community Summaries (kiến trúc Microsoft GraphRAG) giúp truy vấn cấp vĩ mô không bị nghẽn BFS.
