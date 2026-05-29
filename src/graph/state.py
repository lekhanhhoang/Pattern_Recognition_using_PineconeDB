from typing import Annotated, TypedDict, Dict, Any
from langgraph.graph.message import add_messages
from langchain_core.messages import BaseMessage

class AgentState(TypedDict):
    # Lịch sử trò chuyện tự động quản lý bởi LangGraph
    messages: Annotated[list[BaseMessage], add_messages]
    
    # Phân loại ý định của người dùng (vd: tra_cuu_ielts, hoi_dap_ngu_phap, yeu_cau_cham_diem)
    current_intent: str
    
    # Hồ sơ học viên (Mục tiêu điểm Band Score, Kỹ năng yếu, Lịch thi dự kiến)
    student_profile: Dict[str, Any]
    
    # Tác nhân (node) tiếp theo sẽ được gọi trong luồng LangGraph
    next: str