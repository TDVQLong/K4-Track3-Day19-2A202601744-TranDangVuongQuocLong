# 🔍 Báo Cáo Phân Tích Ca Lỗi (Failure Analysis)

**Họ và tên học viên:** Trần Đặng Vương Quốc Long  
**Mã học viên:** 2A202601744  

---

## 1. Ca lỗi Flat RAG thất bại (GraphRAG thành công)
- **Question ID & Câu hỏi:** `G04` — *"Which AI startup founded by former Microsoft engineers received investment from Google, and what product did they release?"*
- **Nguyên nhân Flat RAG thất bại:** 
  - Thông tin xuất thân của sáng lập viên (ex-Microsoft) nằm ở Bài báo A (`art_002`), trong khi tin tức Google rót vốn và sản phẩm `AgentFlow-7B` nằm ở Bài báo B (`art_004`).
  - Vector similarity của câu hỏi chỉ retrieved được các chunk từ Bài báo B do chứa nhiều từ khóa match (`Google`, `investment`, `product`), khiến Flat RAG thiếu context nền tảng về sáng lập viên và trả lời nhầm lẫn/bỏ sót.
- **GraphRAG đã giải quyết như thế nào:** 
  - GraphRAG duyệt đồ thị BFS từ nút seed `Google Ventures` / `Microsoft`:
    `(Google Ventures)-[INVESTED_IN]->(Synthetix AI)-[DEVELOPED]->(AgentFlow-7B)`
  - Tự động kết nối đường đi qua các node trung gian từ 2 bài báo khác nhau, cung cấp đầy đủ thông tin đa hop cho LLM tổng hợp câu trả lời hoàn chỉnh.

---

## 2. Ca lỗi GraphRAG khó khăn / thất bại
- **Question ID & Câu hỏi:** `G05` — *"Identify one technology connected to the same company in at least two news chunks and summarize how the relationship changed over time."*
- **Nguyên nhân:**
  - Việc trích xuất mối quan hệ dòng thời gian từ nhiều chunk bài báo đòi hỏi thuộc tính `published_date` trên các cạnh phải được sắp xếp tuyệt đối.
  - Khi thông tin tên công nghệ ở hai chunk có biến thể từ vựng nhẹ, nếu bước Entity Resolution không gom nhóm chính xác, đồ thị sẽ tách thành 2 nút độc lập làm suy giảm khả năng liên kết temporal graph.
- **Đề xuất khắc phục:** 
  - Bổ sung thuộc tính dòng thời gian (Temporal Valid From/To) trực tiếp trên Nút.
  - Sử dụng thuật toán Time-aware BFS Traversal ưu tiên duyệt theo thứ tự thời gian.
