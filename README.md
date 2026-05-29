# 🎓 Hệ thống Trợ giảng Luyện thi IELTS (AI-Powered)

[![Framework](https://img.shields.io/badge/Framework-LangChain-green)](https://python.langchain.com/)
[![Graph](https://img.shields.io/badge/Orchestration-LangGraph-blue)](https://langchain-ai.github.io/langgraph/)
[![Database](https://img.shields.io/badge/VectorDB-Pinecone-orange)](https://www.pinecone.io/)
[![UI](https://img.shields.io/badge/Frontend-Streamlit-red)](https://streamlit.io/)

## 📖 Giới thiệu
Dự án xây dựng một **AI Agent** chuyên dụng để hỗ trợ ôn luyện và tư vấn thi IELTS. Hệ thống sử dụng kiến trúc **RAG (Retrieval-Augmented Generation)** để cung cấp thông tin học thuật chuẩn xác được cào (crawl) tự động từ trang web của cựu giám khảo bản xứ **IELTS Simon (ielts-simon.study)**.

Hệ thống được thiết kế để hoạt động như một chuyên gia IELTS 8.5+ chuyên nghiệp, hỗ trợ học viên giải đáp các thắc mắc về:
*   Mẹo giải bài thi Reading (True/False/Not Given, Matching Headings...).
*   Cách viết Overview và cấu trúc đoạn văn cho Writing Task 1 & 2.
*   Từ vựng học thuật (Academic Vocabulary) cho Speaking & Listening.
*   Phương pháp học chuẩn tư duy giám khảo chấm thi.

---

## 📋 Mục lục
*   [Giới thiệu](#-giới-thiệu)
*   [Thành viên thực hiện](#-thành-viên-thực-hiện)
*   [Kiến trúc hệ thống](#-kiến-trúc-hệ-thống)
*   [Cấu trúc thư mục](#-cấu-trúc-thư-mục)
*   [Hướng dẫn cài đặt](#-hướng-dẫn-cài-đặt)
*   [Hướng dẫn sử dụng](#-hướng-dẫn-sử-dụng)
*   [Công nghệ sử dụng](#-công-nghệ-sử-dụng)

---

## 👤 Thành viên thực hiện
| Họ tên | Vai trò | Công việc phụ trách |
| :--- | :--- | :--- |
| **Lê Khánh Hoàng** | Trưởng nhóm | Phát triển hệ thống AI, Xây dựng kiến trúc LangGraph, Web Scraping và Pinecone RAG |
| **Trương Xuân Hưng** | Thành viên | Xây dựng nội dung báo cáo Chương 11 và Chương 12 |
| **Lê Quốc Nam** | Thành viên | Xây dựng nội dung báo cáo Chương 1, Chương 2 và Chương 3 |
---

## 🏗️ Kiến trúc hệ thống
Hệ thống được xây dựng dựa trên quy trình **Reasoner-ToolNode** của LangGraph:
1.  **Reasoner (Bộ não)**: Sử dụng mô hình LLM Qwen2.5-72B để phân tích câu hỏi của người dùng.
2.  **Tools (Công cụ)**:
    *   `tra_cuu_tai_lieu_ielts`: Truy xuất dữ liệu bài giảng từ máy chủ Pinecone VectorDB (đã quét 1.972 chunks từ web IELTS Simon).
3.  **State Management**: Quản lý lịch sử hội thoại liên tục bằng LangGraph MemorySaver.

---

## 📂 Cấu trúc thư mục
```text
IELTS-RAG-SYSTEM/
├── src/
│   ├── graph/           # Xây dựng luồng workflow (State, Nodes, Graph) và System Prompt
│   └── tools/           # Các công cụ RAG (ielts_tools.py)
├── frontend1.py         # Giao diện người dùng Streamlit
├── ingest.py            # Script cào web bằng RecursiveUrlLoader và nạp vào Pinecone
├── requirements.txt     # Danh sách thư viện phụ thuộc (đã tối ưu clean code)
└── .env                 # Cấu hình biến môi trường (HF_TOKEN, PINECONE_API_KEY)
```

---

## 🛠️ Hướng dẫn cài đặt

### 1. Chuẩn bị môi trường
Yêu cầu Python 3.10 trở lên. Khuyến khích sử dụng [uv](https://github.com/astral-sh/uv) để quản lý gói nhanh hơn.

### 2. Cài đặt thư viện
```bash
uv pip install -r requirements.txt
```

### 3. Cấu hình biến môi trường
Tạo tệp `.env` tại thư mục gốc và cung cấp các khóa API:
```env
HF_TOKEN=your_huggingface_token_here
PINECONE_API_KEY=your_pinecone_api_key_here
```

---

## 🚀 Hướng dẫn sử dụng

### Bước 1: Nạp dữ liệu IELTS (Crawling & Embedding)
Chạy kịch bản cào dữ liệu từ `ielts-simon.study` và lưu lên Pinecone:
```bash
python ingest.py
```

### Bước 2: Khởi chạy Chatbot
```bash
streamlit run frontend1.py
```

---

## 💻 Công nghệ sử dụng
*   **Ngôn ngữ**: Python
*   **LLM**: Qwen2.5-72B (via Hugging Face API)
*   **Framework**: LangChain, LangGraph
*   **Vector Database**: Pinecone Cloud Database (HuggingFace Embeddings)
*   **Interface**: Streamlit
