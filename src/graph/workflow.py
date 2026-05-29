import os
from dotenv import load_dotenv
from typing import Literal

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

from src.graph.state import AgentState
from src.tools.ielts_tools import ielts_tools

# Load environment variables
load_dotenv()

# 1. Khởi tạo LLM và Bind Tools
hf_token = os.environ.get("HF_TOKEN")
llm = ChatOpenAI(
    base_url="https://router.huggingface.co/v1",
    api_key=hf_token,
    model="Qwen/Qwen2.5-72B-Instruct",
    temperature=0.1
)

# Gắn công cụ vào mô hình
llm_with_tools = llm.bind_tools(ielts_tools)

# 2. Định nghĩa các Node trong Graph

def reasoner(state: AgentState):
    """
    Node xử lý chính: Nhận câu hỏi, áp dụng Persona và quyết định dùng Tool hay trả lời.
    """
    system_prompt = SystemMessage(content=(
        "Bạn là một Chuyên gia Luyện thi IELTS 8.5+ Overall (dựa trên phương pháp của cựu giám khảo Simon).\n"
        "Nhiệm vụ ĐỘC QUYỀN của bạn là tư vấn chuyên môn: cung cấp phương pháp học, giải bài, từ vựng và mẹo thi (Listening, Reading, Writing, Speaking) hoàn toàn dựa trên tập dữ liệu RAG mà bạn tra cứu được.\n"
        "Phong cách: Học thuật, đi thẳng vào vấn đề, dễ hiểu và truyền cảm hứng.\n\n"
        "QUY TẮC TỐI THƯỢNG:\n"
        "1. Bạn BẮT BUỘC phải dùng công cụ tra cứu dữ liệu (Tool) để tìm câu trả lời cho mọi câu hỏi về IELTS. KHÔNG ĐƯỢC tự bịa ra phương pháp.\n"
        "2. CHỈ TRẢ LỜI dựa trên thông tin cào được từ dữ liệu. Nếu người dùng hỏi các thông tin hành chính (như lệ phí thi, cách đăng ký, địa điểm thi tại BC/IDP) hoặc các vấn đề ngoài lề, hãy TRẢ LỜI RÕ RÀNG: 'Tôi chỉ tập trung hỗ trợ chuyên môn ôn luyện IELTS và phương pháp giải đề. Vui lòng liên hệ trực tiếp trang web của IDP/British Council để biết các thông tin thủ tục hành chính.'\n"
        "3. Trình bày phản hồi khoa học, dùng bullet points để người dùng dễ ghi chú."
    ))
    
    # Kết hợp persona và lịch sử hội thoại
    messages = [system_prompt] + state["messages"]
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response]}

# Node chạy Tool tự động
tool_node = ToolNode(ielts_tools)

# 3. Định nghĩa Logic điều hướng (Routing)

def should_continue(state: AgentState) -> Literal["tools", END]:
    """
    Kiểm tra xem LLM có yêu cầu gọi Tool hay không.
    """
    messages = state["messages"]
    last_message = messages[-1]
    
    # Nếu có tool_calls, đi tiếp sang node 'tools'
    if last_message.tool_calls:
        return "tools"
    
    # Nếu không, kết thúc hội thoại
    return END

# 4. Xây dựng Graph

workflow = StateGraph(AgentState)

# Thêm các nút
workflow.add_node("reasoner", reasoner)
workflow.add_node("tools", tool_node)

# Thiết lập điểm bắt đầu
workflow.add_edge(START, "reasoner")

# Thiết lập quan hệ rẽ nhánh từ reasoner
workflow.add_conditional_edges(
    "reasoner",
    should_continue,
    {
        "tools": "tools",
        END: END
    }
)

# Sau khi chạy tool xong, bắt buộc quay lại reasoner để tổng hợp kết quả
workflow.add_edge("tools", "reasoner")

# 5. Biên dịch Workflow
app = workflow.compile(checkpointer=MemorySaver())