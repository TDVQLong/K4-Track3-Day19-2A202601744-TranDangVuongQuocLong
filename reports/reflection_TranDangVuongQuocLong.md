# 💡 Suy Ngẫm & Kế Hoạch Đồ Án (Reflection & Action Plan)

**Học viên:** Trần Đặng Vương Quốc Long  
**Mã học viên:** 2A202601744  

---

## 1. Mapping Bài Giảng vào Code Implementation

| Khái niệm trong bài giảng | Module tương ứng | Hàm / Khối code cụ thể | Quan sát thực tế & Đánh giá |
|--------------------------|------------------|------------------------|-----------------------------|
| **Conservative Coreference** | Module 1 | `resolve_coreference()` | Phân giải đại từ chuẩn xác, ngăn ngừa việc nối sai cạnh (False Edge). |
| **Schema & Allowlist Guard** | Module 2 | `ALLOWED_NODE_TYPES`, `extract_triples()` | Đảm bảo 100% node và relation thu thập tuân thủ strict schema định sẵn. |
| **Bulk Cypher Ingestion** | Module 3 | `run_cypher("UNWIND $rows AS row ...")` | Nạp hàng loạt node/edge vào Neo4j chỉ với 1 query duy nhất, tối ưu I/O. |
| **Entity Resolution & Union-Find** | Module 3 | `DisjointSet`, `lexical_ratio()` | Gom nhóm thực thể trùng lắp minh bạch, lưu vết chi tiết trong bảng audit log. |
| **Super-node Degree Cap** | Module 4 | `retrieve_graph_context()`, `SUPER_NODE_DEGREE` | Cắt tỉa node bậc $> 100$ về $\le 50$ cạnh mới nhất, kiểm soát kích thước context. |
| **LLM-as-a-Judge Evaluation** | Module 5 | `judge_answer()`, `qwen/qwen3.6-27b` | Đánh giá khách quan 3 tiêu chí Comprehensiveness, Faithfulness, Multi-hop. |

---

## 2. Quá Trình Debugging & Bài Học
- **Lỗi kỹ thuật phức tạp nhất:** Lỗi nghẽn Rate Limit (HTTP 429) và lỗi tràn độ dài ngữ cảnh `413 Request Entity Too Large` khi gọi LLM API.
- **Cách xử lý thành công:**
  1. Thêm cơ chế Exponential Backoff với `time.sleep()` tự động tăng dần thời gian chờ khi gặp HTTP 429.
  2. Giới hạn độ dài ngữ cảnh đồ thị hợp lý (`MAX_GRAPH_CONTEXT_CHARS = 14000`, `context[:3500]`).
  3. Cấu hình chuyển đổi mô hình linh hoạt và bóc tách thẻ suy nghĩ `<think>...</think>` bằng Regular Expression.

---

## 3. Kế Hoạch Áp Dụng vào Đồ Án Thực Tế (Action Plan)
- **Tên đồ án:** Hệ thống RAG Phân tích Báo cáo Tài chính & Tin tức Doanh nghiệp Niêm yết (Financial Knowledge RAG).
- **Đặc thù bài toán & Lý do chọn GraphRAG:** Phân tích tài chính doanh nghiệp đòi hỏi kết nối thông tin từ nhiều báo cáo tài chính quý/năm và tin tức M&A rời rạc. Vector RAG thuần túy thường bỏ sót các chuỗi sở hữu chéo (Cross-ownership). GraphRAG giúp truy vết mối quan hệ sở hữu và dòng tiền minh bạch.
- **Cấu trúc Node & Relation dự kiến:**
  - *Nodes:* `Company`, `Executive`, `FinancialMetric`, `Project`, `Event`
  - *Relations:* `ACQUIRED`, `INVESTED_IN`, `APPOINTED_AS`, `REPORTED_METRIC`, `OWNS`
- **Chiến lược xử lý Super-node & Entity Resolution:**
  - Đặt chính sách Super-node Cap cho các tập đoàn lớn (Vingroup, Masan) kết hợp thuộc tính `quarter` và `year` trên các cạnh.
  - Sử dụng Mã Cổ Phiếu (Ticker Symbol - e.g. VIC, VHM, MSN) làm Canonical ID duy nhất trong bước Entity Resolution để triệt tiêu nhập nhằng tên doanh nghiệp.
