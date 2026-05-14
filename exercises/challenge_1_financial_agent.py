"""Challenge 1: Financial Agent

Mở rộng multi-agent system bằng cách thêm financial_agent chuyên phân tích
thiệt hại tài chính, bồi thường, và định giá tổn thất.

Chạy:
    uv run python exercises/challenge_1_financial_agent.py
"""

import asyncio
import os
import sys
from typing import Annotated, TypedDict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from common.llm import get_llm


# ---------------------------------------------------------------------------
# Tools cho financial agent
# ---------------------------------------------------------------------------

@tool
def calculate_damages(scenario: str) -> str:
    """Tính toán ước tính thiệt hại tài chính theo loại vi phạm.

    Args:
        scenario: Mô tả tình huống vi phạm (contract_breach, data_breach, fraud, negligence)
    """
    damage_tables = {
        "contract_breach": (
            "Expectation damages: giá trị hợp đồng đầy đủ. "
            "Consequential damages: lợi nhuận bị mất (cần chứng minh). "
            "Punitive damages: thường không áp dụng cho vi phạm hợp đồng thuần túy. "
            "Liquidated damages: theo điều khoản hợp đồng (nếu có)."
        ),
        "data_breach": (
            "GDPR: phạt tối đa 4% doanh thu toàn cầu hoặc EUR 20M (lấy mức cao hơn). "
            "CCPA: $100-$750/người dùng bị ảnh hưởng cho class action. "
            "Chi phí thông báo vi phạm: $125-$200/người bị ảnh hưởng (trung bình). "
            "Thiệt hại danh tiếng: 1-5% market cap trong 30 ngày sau vụ việc."
        ),
        "fraud": (
            "Compensatory damages: hoàn lại toàn bộ số tiền bị lừa đảo. "
            "Punitive damages: thường gấp 2-3x compensatory damages. "
            "Phạt hình sự theo Wire Fraud Act (18 U.S.C. § 1343): tới $250K/vụ + tù giam. "
            "Restitution: hoàn trả lợi nhuận bất hợp pháp."
        ),
        "negligence": (
            "Special damages: chi phí y tế, thu nhập bị mất (có thể tính toán). "
            "General damages: đau khổ tinh thần, mất khả năng hưởng thụ cuộc sống. "
            "Punitive damages: chỉ khi negligence ở mức reckless/grossly negligent. "
            "Comparative negligence: bồi thường giảm theo % lỗi của nguyên đơn."
        ),
    }

    scenario_lower = scenario.lower()
    for key, text in damage_tables.items():
        if key.replace("_", " ") in scenario_lower or key in scenario_lower:
            return f"[{key}] {text}"

    return (
        "Không xác định được loại thiệt hại cụ thể. "
        "Các loại phổ biến: contract_breach, data_breach, fraud, negligence."
    )


@tool
def estimate_litigation_cost(case_complexity: str) -> str:
    """Ước tính chi phí kiện tụng theo độ phức tạp của vụ án.

    Args:
        case_complexity: Mức độ phức tạp (simple, moderate, complex, mega)
    """
    costs = {
        "simple": (
            "Tranh chấp đơn giản (<$100K): chi phí luật sư $15K-$50K. "
            "Thời gian: 6-12 tháng. Thường giải quyết qua mediation."
        ),
        "moderate": (
            "Tranh chấp trung bình ($100K-$1M): chi phí luật sư $50K-$200K. "
            "Thời gian: 1-2 năm. Expert witnesses: $10K-$50K thêm."
        ),
        "complex": (
            "Tranh chấp phức tạp ($1M-$50M): chi phí luật sư $200K-$2M. "
            "Thời gian: 2-5 năm. E-discovery, depositions, multiple experts."
        ),
        "mega": (
            "Mega litigation (>$50M): chi phí luật sư $2M-$20M+. "
            "Thời gian: 5-10+ năm. Class actions, MDL, international arbitration."
        ),
    }
    return costs.get(
        case_complexity.lower(),
        "Vui lòng chỉ định: simple, moderate, complex, hoặc mega."
    )


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

def _last_wins(left: str | None, right: str | None) -> str:
    return right if right is not None else (left or "")


class State(TypedDict):
    question: str
    law_analysis: Annotated[str, _last_wins]
    tax_analysis: Annotated[str, _last_wins]
    compliance_analysis: Annotated[str, _last_wins]
    financial_analysis: Annotated[str, _last_wins]
    final_response: str


# ---------------------------------------------------------------------------
# Agents
# ---------------------------------------------------------------------------

def law_agent(state: State) -> dict:
    llm = get_llm()
    response = llm.invoke([HumanMessage(content=(
        f"Bạn là luật sư doanh nghiệp cấp cao. Phân tích ngắn gọn (tối đa 150 từ) "
        f"các vấn đề pháp lý trong câu hỏi sau:\n\n{state['question']}"
    ))])
    print(f"  [law_agent] Done ({len(response.content)} chars)")
    return {"law_analysis": response.content}


def check_routing(state: State) -> list[Send]:
    question_lower = state["question"].lower()
    tasks = []

    if any(kw in question_lower for kw in ["tax", "irs", "thuế", "fiscal"]):
        tasks.append(Send("tax_agent", state))

    if any(kw in question_lower for kw in ["compliance", "sec", "regulation", "gdpr"]):
        tasks.append(Send("compliance_agent", state))

    if any(kw in question_lower for kw in [
        "financial", "damages", "compensation", "thiệt hại", "bồi thường",
        "tổn thất", "phạt", "penalty", "fine", "cost", "chi phí",
    ]):
        tasks.append(Send("financial_agent", state))

    return tasks if tasks else [Send("aggregate_results", state)]


def tax_agent(state: State) -> dict:
    llm = get_llm()
    response = llm.invoke([HumanMessage(content=(
        f"Bạn là chuyên gia thuế. Phân tích ngắn gọn (tối đa 100 từ) khía cạnh thuế:\n\n"
        f"Câu hỏi: {state['question']}\nPhân tích pháp lý: {state.get('law_analysis', 'N/A')}"
    ))])
    print(f"  [tax_agent] Done ({len(response.content)} chars)")
    return {"tax_analysis": response.content}


def compliance_agent(state: State) -> dict:
    llm = get_llm()
    response = llm.invoke([HumanMessage(content=(
        f"Bạn là chuyên gia compliance. Phân tích ngắn gọn (tối đa 100 từ) khía cạnh tuân thủ:\n\n"
        f"Câu hỏi: {state['question']}\nPhân tích pháp lý: {state.get('law_analysis', 'N/A')}"
    ))])
    print(f"  [compliance_agent] Done ({len(response.content)} chars)")
    return {"compliance_analysis": response.content}


async def financial_agent(state: State) -> dict:
    """Financial agent dùng ReAct pattern với tools tính toán thiệt hại."""
    from langgraph.prebuilt import create_react_agent

    print("  [financial_agent] Analysing financial damages...")
    llm = get_llm()
    agent = create_react_agent(
        model=llm,
        tools=[calculate_damages, estimate_litigation_cost],
        prompt=(
            "Bạn là chuyên gia tài chính pháp lý (forensic accountant & litigation economist). "
            "Sử dụng tools để ước tính thiệt hại tài chính và chi phí kiện tụng. "
            "Trả lời ngắn gọn, có số liệu cụ thể, tối đa 150 từ."
        ),
    )
    result = await agent.ainvoke({"messages": [{"role": "user", "content": state["question"]}]})
    final_msg = result["messages"][-1].content
    print(f"  [financial_agent] Done ({len(final_msg)} chars)")
    return {"financial_analysis": final_msg}


def aggregate_results(state: State) -> dict:
    llm = get_llm()
    sections = []
    if state.get("law_analysis"):
        sections.append(f"## Phân Tích Pháp Lý\n{state['law_analysis']}")
    if state.get("tax_analysis"):
        sections.append(f"## Phân Tích Thuế\n{state['tax_analysis']}")
    if state.get("compliance_analysis"):
        sections.append(f"## Phân Tích Tuân Thủ\n{state['compliance_analysis']}")
    if state.get("financial_analysis"):
        sections.append(f"## Phân Tích Thiệt Hại Tài Chính\n{state['financial_analysis']}")

    combined = "\n\n---\n\n".join(sections)
    response = llm.invoke([HumanMessage(content=(
        f"Tổng hợp các phân tích sau thành báo cáo pháp lý hoàn chỉnh, "
        f"ngắn gọn (<400 từ), có cấu trúc rõ ràng:\n\n{combined}\n\n"
        f"Câu hỏi gốc: {state['question']}"
    ))])
    print(f"  [aggregate_results] Done ({len(response.content)} chars)")
    return {"final_response": response.content}


# ---------------------------------------------------------------------------
# Graph
# ---------------------------------------------------------------------------

def build_graph():
    graph = StateGraph(State)
    graph.add_node("law_agent", law_agent)
    graph.add_node("check_routing", check_routing)
    graph.add_node("tax_agent", tax_agent)
    graph.add_node("compliance_agent", compliance_agent)
    graph.add_node("financial_agent", financial_agent)
    graph.add_node("aggregate_results", aggregate_results)

    graph.add_edge(START, "law_agent")
    graph.add_edge("law_agent", "check_routing")
    graph.add_conditional_edges("check_routing", lambda x: x)
    graph.add_edge("tax_agent", "aggregate_results")
    graph.add_edge("compliance_agent", "aggregate_results")
    graph.add_edge("financial_agent", "aggregate_results")
    graph.add_edge("aggregate_results", END)

    return graph.compile()


QUESTION = (
    "Công ty bị kiện vì vi phạm hợp đồng và gian lận tài chính. "
    "Thiệt hại ước tính là gì và chi phí kiện tụng sẽ như thế nào?"
)


async def main():
    load_dotenv()
    print("=" * 70)
    print("CHALLENGE 1: Multi-Agent System + Financial Agent")
    print("=" * 70)
    print()
    print("[Kiến trúc]")
    print("  law_agent → check_routing → [tax_agent | compliance_agent | financial_agent]")
    print("                           → aggregate_results → END")
    print()
    print(f"Câu hỏi: {QUESTION}")
    print("-" * 70)

    graph = build_graph()
    result = await graph.ainvoke({
        "question": QUESTION,
        "law_analysis": "",
        "tax_analysis": "",
        "compliance_analysis": "",
        "financial_analysis": "",
        "final_response": "",
    })

    print("\n" + "=" * 70)
    print("KẾT QUẢ CUỐI CÙNG")
    print("=" * 70)
    print(result["final_response"])
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
