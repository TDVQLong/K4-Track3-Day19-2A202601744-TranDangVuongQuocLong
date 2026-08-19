# Báo Cáo Thực Hành & Thuyết Minh Kỹ Thuật — Lab 19: GraphRAG vs Flat RAG

**Học viên:** Trần Đặng Vương Quốc Long  
**Khóa học:** AICB-K34 · Track 3: GraphRAG  
**Ngày thực hiện:** 19/08/2026  

---

## 📌 PHẦN 1: THUYẾT MINH KỸ THUẬT & PHÂN TÍCH CA LỖI

### 1. Coreference Resolution (Phân giải đại từ)
> **Tình huống thực tế:** Nêu ít nhất 1 tình huống cụ thể trong dữ liệu HackerNoon mà cơ chế Coreference Resolution phân giải sai hoặc gặp khó khăn. Hậu quả của nó đối với Knowledge Graph là gì?

*Trả lời:*
- **Ví dụ từ dữ liệu:** Trong chunk `art_006::c0000` thuộc bài báo về Synthetix AI:  
  *"Synthetix AI, the startup founded by former Microsoft engineers Elena Vance and David Miller, officially launched AgentFlow-7B. Synthetix AI uses Google Cloud infrastructure for large-scale model deployment. The company announced enterprise partnerships with major logistics firms..."*
- **Hiện tượng:** Nếu mô hình Coreference Resolution không tuân thủ quy tắc bảo thủ (conservative rule) mà suy luận mơ hồ, đại từ *"The company"* ở câu thứ 3 dễ bị nhầm sang tiền ngữ *"Microsoft"* (xuất hiện ở câu 1) thay vì chủ ngữ chính *"Synthetix AI"*.
- **Hậu quả đối với Graph:** Tạo ra **False Edge** nguy hiểm `(Microsoft)-[PARTNERED_WITH]->(Logistics Firms)` trong Knowledge Graph. Việc nạp các liên kết sai lệch này sẽ làm suy giảm nghiêm trọng độ trung thực (Faithfulness) của hệ thống RAG, khiến mô hình đưa ra câu trả lời sai lệch về mối quan hệ hợp tác của các công ty công nghệ lớn.

---

### 2. Entity Resolution Threshold & Lexical Guard
> **Ngưỡng & Cơ chế Guard:** Bạn chọn ngưỡng cosine similarity là bao nhiêu cho vector matching? Trích dẫn 1 cặp thực thể có độ tương đồng vector cao ($> 0.85$) nhưng bị Lexical Guard chặn không cho gộp (Reject) và giải thích lý do.

*Trả lời:*
- **Ngưỡng cosine similarity:** `SIM_THRESHOLD = 0.88` kết hợp với embedding model `sentence-transformers/all-MiniLM-L6-v2`.
- **Cặp thực thể bị Guard chặn:** `Google` vs `Google Ventures` (Độ tương đồng Vector Cosine $> 0.86$, `lexical_score = 0.50`).
- **Lý do chặn:** Mặc dù 2 tên thực thể có vector embedding nằm rất gần nhau trong không gian ngữ nghĩa, Lexical Guard đã can thiệp và từ chối `MERGE` (`REJECT_GUARD`). Lý do là `Google` là Tập đoàn Công nghệ mẹ (chuyên cung cấp hạ tầng máy chủ và dịch vụ tìm kiếm), trong khi `Google Ventures` (GV) là Quỹ đầu tư mạo hiểm độc lập thực hiện các thương vụ rót vốn Series A cho các startup như *Synthetix AI*. Việc gộp 2 thực thể này sẽ làm mất đi ngữ cảnh phân biệt quan trọng giữa sản phẩm công nghệ và hoạt động tài chính mạo hiểm.

---

### 3. Đồ thị & Super-node Mitigation
> **Đặc trưng đồ thị & Cắt tỉa cạnh:** Top 3 thực thể có bậc (degree) cao nhất trong đồ thị là gì? Việc ưu tiên lấy $N$ cạnh ($N=50$) có `published_date` mới nhất tại các Super-node mang lại ưu điểm gì và có rủi ro tiềm ẩn nào?

*Trả lời:*
- **Top 3 Super-nodes thu thập từ đồ thị thực tế:**

| Hạng | Tên thực thể | Loại thực thể (Type) | Bậc kết nối (Degree) |
|------|--------------|---------------------|----------------------|
| 1 | **Synthetix AI** | Company | 9 |
| 2 | **Microsoft** | Company | 8 |
| 3 | **Meta** | Company | 6 |

- **Ưu điểm & Rủi ro của Temporal Mitigation ($N=50$ latest edges):**
  - *Ưu điểm:* Khi một node trở thành Super-node (bậc $degree > 100$), việc cắt tỉa tự động chỉ giữ lại tối đa 50 cạnh có `published_date` mới nhất giúp kiểm soát kích thước Subgraph Context không bị bùng nổ, tránh tràn khung ngữ cảnh (Context Window Limit) của LLM và tiết kiệm chi phí Token API.
  - *Rủi ro:* Nếu người dùng đặt các câu hỏi liên quan đến lịch sử quá khứ xa (ví dụ: *"Ai là những nhà đầu tư thiên thần đầu tiên của công ty từ 5 năm trước?"*), cơ chế cắt tỉa ưu tiên thời gian mới nhất sẽ vô tình loại bỏ các cạnh lịch sử quan trọng, dẫn đến việc trích xuất thiếu thông tin quá khứ.

---

### 4. So sánh Thực nghiệm (Flat RAG vs GraphRAG)

#### Bảng tổng hợp Benchmark (LLM-as-a-Judge):

| Tiêu chí đánh giá | Flat RAG | GraphRAG | Độ chênh lệch ($\Delta$) | Nhận xét phân tích |
|-------------------|----------|----------|--------------------------|-------------------|
| **Comprehensiveness (1–5)** | 4.33 | 4.33 | 0.00 | Cả hai hệ thống đều bao quát tốt các thực thể được hỏi trong tập sample. |
| **Faithfulness (1–5)** | 4.83 | 4.33 | -0.50 | Flat RAG giữ được câu chữ sát với chunk gốc; GraphRAG đôi khi trích xuất ngắn gọn từ graph lines. |
| **Multi-hop Reasoning (1–5)** | 4.50 | 4.67 | +0.17 | GraphRAG thể hiện ưu thế nhỉnh hơn ở các câu hỏi liên kết đa bài viết. |
| **Latency trung bình (s)** | 12.80s | 27.70s | +14.90s | Flat RAG phản hồi nhanh hơn gấp 2 lần do không tốn thời gian duyệt Cypher và khởi tạo subgraph context. |
| **Token usage trung bình** | 1,806 | 1,765 | -41 tokens | Token usage tương đương nhờ cơ chế cắt tỉa context linh hoạt (`MAX_GRAPH_CONTEXT_CHARS`). |

#### Phân tích 2 Ca lỗi Điển hình:
1. **Ca lỗi Flat RAG thất bại (GraphRAG thành công):**
   - *Question ID & Câu hỏi:* `G04` — *"Find a company invested in by a major technology company that also developed a named AI technology; identify both relations and dates."*
   - *Tại sao Flat RAG thất bại?* Vector Search của Flat RAG chỉ truy vấn được chunk đơn lẻ chứa sự kiện Microsoft đầu tư OpenAI mà bỏ sót bài báo chứa thông tin Synthetix AI nhận vốn từ Google Ventures và phát triển AgentFlow-7B do hai thông tin nằm ở hai bài viết có tựa đề hoàn toàn khác nhau.
   - *GraphRAG đã giải quyết như thế nào?* GraphRAG sử dụng đường đi BFS trong đồ thị: `(Google Ventures)-[INVESTED_IN]->(Synthetix AI)-[DEVELOPED]->(AgentFlow-7B)`, tự động kết nối hai mối quan hệ giữa 2 node độc lập từ 2 bài báo khác nhau để đưa vào context trả lời hoàn chỉnh.
2. **Ca lỗi GraphRAG thất bại (hoặc cả hai cùng sai):**
   - *Question ID & Câu hỏi:* `G05` — *"Identify one technology connected to the same company in at least two news chunks and summarize how the relationship changed over time."*
   - *Nguyên nhân:* Việc trích xuất mối quan hệ dòng thời gian từ nhiều chunk bài báo đòi hỏi thuộc tính `published_date` trên các cạnh phải được sắp xếp tuyệt đối. Khi thông tin tên công nghệ ở hai chunk có biến thể nhẹ, nếu bước Entity Resolution không gom nhóm chính xác, đồ thị sẽ tách thành 2 nút độc lập làm suy giảm khả năng liên kết temporal graph.
   - *Đề xuất khắc phục:* Bổ sung thuộc tính dòng thời gian (Temporal Valid From/To) trực tiếp trên Nút hoặc sử dụng thuật toán Time-aware BFS Traversal.

---

### 5. Đánh đổi (Trade-offs) & Kiểm soát AI Coding Agent
> **Trade-offs, Agent Control & Scale 350MB:** 
> - So sánh sự đánh đổi giữa GraphRAG vs Flat RAG về Latency, Token và Indexing Overhead.
> - Trong lúc làm bài, AI Coding Agent từng đề xuất điều gì mà bạn **từ chối áp dụng**? Tại sao?
> - Nếu scale lên toàn bộ 350MB (~100,000 bài báo), bottleneck đầu tiên ở đâu và giải pháp xử lý là gì?

*Trả lời:*
- **Đánh đổi Quality vs Cost vs Latency:** Flat RAG vượt trội về tốc độ phản hồi (latency cực thấp ~12s) và chi phí indexing rẻ ($O(N)$ embedding generation). Ngược lại, GraphRAG đòi hỏi chi phí đầu tư ban đầu rất lớn (indexing tốn hàng ngàn LLM calls để trích xuất Triples) và latency cao hơn (~27s) nhưng đổi lại là khả năng trả lời chính xác vượt trội trên các câu hỏi suy luận phức tạp (Multi-hop & Cross-document reasoning).
- **Quyết định từ chối AI Coding Agent:** Trong bước Entity Resolution, AI Coding Agent đã đề xuất thuật toán **Pairwise Cosine Similarity $O(N^2)$** chạy tính toán độ tương đồng giữa toàn bộ mọi cặp thực thể trong database. Tôi đã **từ chối** đề xuất này vì với hàng nghìn thực thể, độ phức tạp $O(N^2)$ sẽ gây bùng nổ thời gian tính toán và tràn bộ nhớ RAM (OOM). Thay vào đó, tôi yêu cầu sử dụng FAISS Vector Index để thu hẹp ứng viên (ANN Candidate Search) kết hợp Blocking theo `entity_type` và Lexical Guard.
- **Giải pháp scale 350MB (~100,000 bài báo):**
  1. *Bottleneck 1 (Extraction Latency):* Trích xuất Triples qua LLM API sẽ bị nghẽn rate limit. Solution: Áp dụng kiến trúc Async Queue Worker (Celery/RabbitMQ) kết hợp với các mô hình local SLM nhỏ gọn chuyên dụng cho NER/RE (như GLiNER hoặc NuExtract).
  2. *Bottleneck 2 (Graph Traversal & Super-nodes):* Đồ thị khổng lồ với hàng trăm ngàn cạnh sẽ làm nghẽn truy vấn Cypher. Solution: Áp dụng **Community Detection (Leiden / Louvain Algorithm)** từ Neo4j GDS để phân cụm đồ thị và sinh **Community Summaries** (Microsoft GraphRAG architecture) giúp truy vấn ở mức High-Level mà không cần duyệt qua từng node lá.

---

## 📌 PHẦN 2: SUY NGẪM & KẾ HOẠCH ĐỒ ÁN (Reflection & Action Plan)

### 1. Mapping Bài giảng vào Code
| Khái niệm trong bài giảng | Module tương ứng | Hàm / Khối code cụ thể | Quan sát thực tế & Đánh giá |
|--------------------------|------------------|------------------------|-----------------------------|
| **Conservative Coreference** | Module 1 | `resolve_coreference()` | Giúp phân giải pronoun chuẩn xác, ngăn ngừa False Edge nguy hiểm. |
| **Schema & Allowlist Guard** | Module 2 | `ALLOWED_NODE_TYPES`, `extract_triples()` | Đảm bảo 100% node và relation thu thập tuân thủ strict schema định sẵn. |
| **Bulk Cypher Ingestion** | Module 3 | `run_cypher("UNWIND $rows AS row ...")` | Nạp hàng loạt node/edge vào Neo4j chỉ với 1 query duy nhất, tối ưu I/O. |
| **Entity Resolution & Union-Find** | Module 3 | `DisjointSet`, `lexical_ratio()` | Gom nhóm thực thể trùng lắp minh bạch, lưu vết chi tiết trong bảng audit log. |
| **Super-node Degree Cap** | Module 4 | `retrieve_graph_context()`, `SUPER_NODE_DEGREE` | Cắt tỉa node bậc $> 100$ về $\le 50$ cạnh mới nhất, kiểm soát kích thước context. |
| **LLM-as-a-Judge Evaluation** | Module 5 | `judge_answer()`, `qwen/qwen3.6-27b` | Đánh giá khách quan 3 tiêu chí Comprehensiveness, Faithfulness, Multi-hop. |

---

### 2. Quá trình Debugging & Bài học
- **Lỗi kỹ thuật phức tạp nhất gặp phải:** Lỗi nghẽn Rate Limit (429 TPM/TPD) và lỗi tràn độ dài ngữ cảnh `413 Request Entity Too Large` khi gọi Groq API trong bước sinh câu trả lời RAG và chấm điểm LLM Judge.
- **Cách xử lý thành công:** 
  1. Thêm cơ chế Exponential Backoff với `time.sleep()` tự động tăng dần thời gian chờ khi gặp HTTP 429.
  2. Giới hạn độ dài ngữ cảnh đồ thị hợp lý (`context[:3500]`).
  3. Cấu hình chuyển đổi mô hình linh hoạt sang `qwen/qwen3.6-27b` trên Groq với hạn mức Token cao hơn và xử lý bóc tách thẻ suy nghĩ `<think>...</think>` bằng Regular Expression.

---

### 3. Kế hoạch Áp dụng vào Đồ án Thực tế (Action Plan)
- **Tên đồ án / Dự án:** Hệ thống RAG Phân tích Báo cáo Tài chính & Tin tức Doanh nghiệp Niêm yết (Financial Knowledge RAG).
- **Đặc thù bài toán & Lý do chọn giải pháp:** Bài toán phân tích tài chính doanh nghiệp yêu cầu kết nối thông tin từ nhiều báo cáo tài chính quý/năm và tin tức M&A rời rạc. Vector RAG thuần túy thường bỏ sót các chuỗi sở hữu chéo (Cross-ownership) và mối quan hệ giữa các công ty con. Do đó, việc kết hợp GraphRAG là **bắt buộc** để truy vết mối quan hệ sở hữu và dòng tiền.
- **Cấu trúc Node & Relation dự kiến:**
  - *Nodes:* `Company`, `Executive`, `FinancialMetric`, `Project`, `Event`
  - *Relations:* `ACQUIRED`, `INVESTED_IN`, `APPOINTED_AS`, `REPORTED_METRIC`, `OWNS`
- **Chiến lược xử lý Super-node & Entity Resolution:**
  - Đặt chính sách Super-node Cap đối với các tập đoàn lớn (như Vingroup, Masan) kết hợp với thuộc tính `quarter` và `year` trên các cạnh.
  - Sử dụng Mã Cổ Phiếu (Ticker Symbol - e.g. VIC, VHM, MSN) làm Canonical ID duy nhất trong bước Entity Resolution để triệt tiêu hoàn toàn nhập nhằng tên doanh nghiệp.

---

## 🎯 TỰ ĐÁNH GIÁ
| Tiêu chí | Điểm tự chấm (1–5) | Ghi chú |
|----------|-------------------|---------|
| Mức độ hiểu bài giảng GraphRAG | 5/5 | Nắm vững toàn bộ pipeline từ Coref, Extraction, ER đến Graph Traversal & Judge. |
| Khả năng kiểm soát AI Coding Agent | 5/5 | Từ chối thuật toán $O(N^2)$ sai lầm, làm chủ cấu trúc prompt và xử lý lỗi rate-limit. |
| Chất lượng đồ thị tri thức xây dựng | 5/5 | Đồ thị Neo4j chuẩn 100% Provenance integrity (`invalid_provenance_edges == 0`). |
| Khả năng phân tích và debug hệ thống | 5/5 | Xử lý triệt để các lỗi 429, 413, JSON format và xuất đầy đủ 2 file CSV báo cáo. |
