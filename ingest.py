import os
import time
from dotenv import load_dotenv
from langchain_community.document_loaders import RecursiveUrlLoader
from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore

# Load environment variables
load_dotenv()

def bs4_extractor(html: str) -> str:
    """Hàm giải mã HTML, lấy text thuần"""
    soup = BeautifulSoup(html, "lxml")
    
    # Xóa các tag không cần thiết như script, style, nav, footer
    for script in soup(["script", "style", "nav", "footer", "header", "aside"]):
        script.decompose()
    return soup.get_text(separator="\n", strip=True)

def ingest_ielts_data(url: str = "https://ielts-simon.study", index_name: str = "ielts-index", namespace: str = "ielts-prep"):
    """
    Crawl dữ liệu từ website luyện thi IELTS và lưu vào Pinecone VectorDB.
    """
    pinecone_api_key = os.environ.get("PINECONE_API_KEY")
    if not pinecone_api_key:
        print("⚠️ Lỗi: Không tìm thấy PINECONE_API_KEY trong file .env")
        return

    print("🚀 Đang kết nối tới Pinecone...")
    pc = Pinecone(api_key=pinecone_api_key)

    if index_name not in pc.list_indexes().names():
        print(f"🌟 Đang tự động tạo Serverless Index mới tên là '{index_name}' (Dimension: 384, Metric: cosine)...")
        pc.create_index(
            name=index_name,
            dimension=384, 
            metric='cosine',
            spec=ServerlessSpec(
                cloud='aws',
                region='us-east-1'
            )
        )
        while not pc.describe_index(index_name).status['ready']:
            time.sleep(1)
        print("✅ Đã tạo Index thành công!")
    else:
        print(f"✅ Tìm thấy Index '{index_name}' trên Pinecone.")

    print(f"🚀 Bắt đầu cào HTML kiến thức IELTS từ website: {url}...")
    # Bước 1: Cào văn bản HTML
    html_loader = RecursiveUrlLoader(
        url=url,
        max_depth=3, # Crawl sâu 3 cấp độ
        extractor=bs4_extractor,
        prevent_outside=True # Chỉ lấy trong domain chỉ định
    )
    all_data = html_loader.load()
    print(f"✅ Đã tải HTML từ {len(all_data)} bài viết/trang web.")

    if not all_data:
        print("⚠️ Không có dữ liệu nào để nạp.")
        return

    # Bước 2: Chia nhỏ tài liệu (Chunking)
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    all_splits = text_splitter.split_documents(all_data)
    
    # LỌC METADATA: Xóa các giá trị None để tránh lỗi 400 Pinecone
    for doc in all_splits:
        doc.metadata = {k: str(v) for k, v in doc.metadata.items() if v is not None}

    print(f"✅ Đã chia nhỏ tổng cộng thành {len(all_splits)} đoạn văn bản kiến thức IELTS.")

    # Bước 3: Tạo Embeddings
    print("🧠 Đang tạo embeddings (sử dụng HuggingFace)...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Bước 4: Đẩy lên Pinecone
    print(f"🧹 Đang xóa dữ liệu cũ (nếu có) trong Namespace '{namespace}' để làm sạch...")
    try:
        pc.Index(index_name).delete(delete_all=True, namespace=namespace)
        time.sleep(2)
    except Exception as e:
        pass

    print(f"💾 Đang lưu dữ liệu IELTS lên Pinecone vào Namespace: '{namespace}'...")
    PineconeVectorStore.from_documents(
        documents=all_splits,
        embedding=embeddings,
        index_name=index_name,
        namespace=namespace
    )
    
    print("✨ Hoàn tất! Toàn bộ kiến thức IELTS đã sẵn sàng trên Pinecone.")

if __name__ == "__main__":
    IELTS_URL = "https://ielts-simon.study"
    ingest_ielts_data(IELTS_URL)
