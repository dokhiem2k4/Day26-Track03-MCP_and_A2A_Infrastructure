"""Challenge 2: Conversation Memory

Agent nhớ lịch sử các câu hỏi trước đó trong cùng một session.
Mỗi câu hỏi mới được trả lời có xét đến ngữ cảnh từ các câu hỏi cũ.

Điểm mấu chốt:
- State chứa `conversation_history: list[str]` — tích lũy qua các turns
- Reducer `_append` nối thêm vào list thay vì ghi đè
- law_agent nhận toàn bộ history làm ngữ cảnh

Chạy:
    uv run python exercises/challenge_2_memory.py
"""

import asyncio
import os
import sys
from typing import Annotated, TypedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from common.llm import get_llm


# ---------------------------------------------------------------------------
# Reducer: nối thêm vào list (không ghi đè)
# ---------------------------------------------------------------------------

def _append(left: list, right: list) -> list:
    """Reducer: append right list vào left list."""
    return (left or []) + (right or [])


def _last_wins(left: str | None, right: str | None) -> str:
    return right if right is not None else (left or "")


# ---------------------------------------------------------------------------
# State với conversation history
# ---------------------------------------------------------------------------

class State(TypedDict):
    question: str
    conversation_history: Annotated[list[str], _append]  # tích lũy qua turns
    law_analysis: Annotated[str, _last_wins]
    final_response: str


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

def law_agent(state: State) -> dict:
    """Phân tích pháp lý có xét ngữ cảnh lịch sử hội thoại."""
    llm = get_llm()

    history = state.get("conversation_history", [])
    history_text = ""
    if history:
        history_text = "\n\n**Lịch sử hội thoại trước đó:**\n" + "\n".join(
            f"  Turn {i + 1}: {entry}" for i, entry in enumerate(history)
        )

    messages = [
        SystemMessage(content=(
            "Bạn là luật sư doanh nghiệp cấp cao chuyên về luật hợp đồng và doanh nghiệp. "
            "Nếu có lịch sử hội thoại, hãy dựa vào ngữ cảnh đó để trả lời liền mạch hơn. "
            "Tối đa 200 từ."
        )),
        HumanMessage(content=f"{history_text}\n\n**Câu hỏi hiện tại:** {state['question']}"),
    ]

    response = llm.invoke(messages)
    print(f"  [law_agent] Done ({len(response.content)} chars)")
    return {"law_analysis": response.content}


def aggregate_and_save(state: State) -> dict:
    """Tổng hợp và lưu turn hiện tại vào history."""
    llm = get_llm()

    response = llm.invoke([HumanMessage(content=(
        f"Tóm tắt phân tích pháp lý sau thành câu trả lời hoàn chỉnh (<150 từ):\n\n"
        f"{state['law_analysis']}\n\nCâu hỏi gốc: {state['question']}"
    ))])

    final = response.content

    # Lưu turn này vào history cho lần sau
    history_entry = f"Q: {state['question']} | A: {final[:100]}..."

    print(f"  [aggregate] Done. Saved to history.")
    return {
        "final_response": final,
        "conversation_history": [history_entry],
    }


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(State)
    graph.add_node("law_agent", law_agent)
    graph.add_node("aggregate_and_save", aggregate_and_save)

    graph.add_edge(START, "law_agent")
    graph.add_edge("law_agent", "aggregate_and_save")
    graph.add_edge("aggregate_and_save", END)

    return graph.compile()


# ---------------------------------------------------------------------------
# Multi-turn conversation loop
# ---------------------------------------------------------------------------

async def main():
    load_dotenv()

    print("=" * 70)
    print("CHALLENGE 2: Conversation Memory")
    print("=" * 70)
    print()
    print("[Cách hoạt động]")
    print("  - Mỗi câu hỏi được thêm vào lịch sử sau khi trả lời")
    print("  - Agent biết ngữ cảnh từ các câu hỏi trước")
    print("  - State tích lũy history qua _append reducer")
    print()

    graph = build_graph()

    # Lịch sử tích lũy giữa các turns
    conversation_history: list[str] = []

    questions = [
        "Hợp đồng mua bán có cần công chứng không?",
        "Nếu một bên vi phạm hợp đồng vừa nói, hậu quả pháp lý là gì?",
        "Thời hiệu để khởi kiện trong trường hợp này là bao lâu?",
    ]

    for i, question in enumerate(questions, 1):
        print(f"\n{'=' * 70}")
        print(f"TURN {i}: {question}")
        print("=" * 70)
        print(f"  History có sẵn: {len(conversation_history)} entries")

        result = await graph.ainvoke({
            "question": question,
            "conversation_history": conversation_history,
            "law_analysis": "",
            "final_response": "",
        })

        print(f"\n[TRẢ LỜI TURN {i}]")
        print(result["final_response"])

        # Cập nhật history cho turn tiếp theo
        conversation_history = result["conversation_history"]

    print("\n" + "=" * 70)
    print("TOÀN BỘ LỊCH SỬ HỘI THOẠI")
    print("=" * 70)
    for i, entry in enumerate(conversation_history, 1):
        print(f"  {i}. {entry}")
    print()
    print("[Điểm học] State.conversation_history dùng _append reducer")
    print("  nên mỗi turn THÊM VÀO thay vì ghi đè lịch sử cũ.")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
