import os
from langchain_core.tools import tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_pinecone import PineconeVectorStore

# Cấu hình Pinecone cho dự án IELTS
PINECONE_INDEX_NAME = "ielts-index"
PINECONE_NAMESPACE = "ielts-prep"

@tool
def tra_cuu_tai_lieu_ielts(query: str) -> str:
    """
    Tra cứu kinh nghiệm, mẹo luyện thi, từ vựng và ngữ pháp IELTS.
    Sử dụng công cụ này bất cứ khi nào người dùng hỏi về cách học tiếng Anh hoặc thi IELTS.
    """
    try:
        pinecone_api_key = os.environ.get("PINECONE_API_KEY")
        if not pinecone_api_key:
            return "Lỗi: Không tìm thấy PINECONE_API_KEY. Vui lòng cấu hình file .env."

        # Khởi tạo Embeddings Model
        embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        
        vectorstore = PineconeVectorStore(
            index_name=PINECONE_INDEX_NAME, 
            embedding=embeddings, 
            namespace=PINECONE_NAMESPACE
        )
        
        docs = vectorstore.similarity_search(query, k=5)
        
        if not docs:
            return "Không tìm thấy thông tin liên quan trong thư viện tài liệu IELTS."
            
        return "\n\n".join([doc.page_content for doc in docs])
    except Exception as e:
        return f"Lỗi khi tra cứu thông tin từ Pinecone: {str(e)}"

# Danh sách các tool để đăng ký với Agent
ielts_tools = [tra_cuu_tai_lieu_ielts]
