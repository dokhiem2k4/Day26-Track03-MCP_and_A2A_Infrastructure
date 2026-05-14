"""Challenge 3: Custom Tool gọi API thực

Tạo tool gọi REST API bên ngoài để tra cứu thông tin pháp lý.
Demo này dùng 2 nguồn thực tế không cần auth:
  - api.fiscozen.it/articles — luật kinh doanh Ý (public)  [fallback: mock]
  - api.law.cornell.edu/v1/search — Cornell LII legal search [fallback: mock]

Trong thực tế production sẽ dùng: LexisNexis API, Westlaw API, hoặc
database nội bộ của công ty. Pattern HTTP async là giống hệt nhau.

Chạy:
    uv run python exercises/challenge_3_custom_tool.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import httpx
from dotenv import load_dotenv
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent

from common.llm import get_llm


# ---------------------------------------------------------------------------
# Tool gọi API thực (với fallback mock nếu API không khả dụng)
# ---------------------------------------------------------------------------

_MOCK_LEGAL_DB = {
    "breach of contract": (
        "[Cornell LII] Breach of contract occurs when a party fails to fulfill "
        "their obligations under a valid contract. Remedies include: compensatory "
        "damages, specific performance, rescission, and restitution. "
        "See UCC § 2-711 to § 2-725 for goods contracts."
    ),
    "tax evasion": (
        "[IRS.gov] Tax evasion (26 U.S.C. § 7201) is a felony punishable by up to "
        "5 years imprisonment and $250,000 fine. Distinct from tax avoidance (legal). "
        "Criminal referrals require willful intent to evade."
    ),
    "gdpr violation": (
        "[EUR-Lex] GDPR Article 83 establishes a two-tier fine structure: "
        "Tier 1 (up to EUR 10M or 2% global turnover): procedural violations. "
        "Tier 2 (up to EUR 20M or 4% global turnover): substantive violations "
        "including unlawful processing and data subject rights."
    ),
    "negligence": (
        "[Restatement Third of Torts] Negligence requires: (1) duty of care, "
        "(2) breach of that duty, (3) causation (actual and proximate), "
        "(4) damages. Contributory/comparative negligence may reduce recovery."
    ),
}


async def _fetch_legal_api(query: str, timeout: float = 5.0) -> str | None:
    """Gọi public legal search API. Trả về None nếu API không khả dụng."""
    url = "https://api.case.law/v1/cases/"
    params = {"search": query, "page_size": 1, "full_case": "false"}
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.get(url, params=params)
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("results", [])
                if results:
                    case = results[0]
                    return (
                        f"[Case Law API] {case.get('name', 'N/A')} "
                        f"({case.get('decision_date', 'N/A')}) — "
                        f"Court: {case.get('court', {}).get('name', 'N/A')}. "
                        f"Jurisdiction: {case.get('jurisdiction', {}).get('name', 'N/A')}."
                    )
    except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPError):
        pass
    return None


@tool
async def search_legal_database(query: str) -> str:
    """Tra cứu cơ sở dữ liệu pháp lý để tìm án lệ và quy định liên quan.

    Args:
        query: Từ khóa pháp lý cần tra cứu (tiếng Anh cho kết quả tốt nhất)
    """
    # Thử gọi API thực trước
    api_result = await _fetch_legal_api(query)
    if api_result:
        return api_result

    # Fallback: mock database khi API không khả dụng
    query_lower = query.lower()
    for key, text in _MOCK_LEGAL_DB.items():
        if any(word in query_lower for word in key.split()):
            return f"[Mock DB — API unavailable] {text}"

    return (
        "[Mock DB] No specific match found. "
        "Try keywords: breach of contract, tax evasion, gdpr violation, negligence."
    )


@tool
async def get_statute_text(statute_code: str) -> str:
    """Lấy nội dung điều luật từ cơ sở dữ liệu pháp lý.

    Args:
        statute_code: Mã điều luật (ví dụ: '26 USC 7201', 'UCC 2-725', 'GDPR Art 83')
    """
    # Gọi Cornell LII API (public, không cần auth)
    code_clean = statute_code.strip().upper()
    url = f"https://api.law.cornell.edu/v1/usc/{code_clean.replace(' ', '/')}"

    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            resp = await client.get(url)
            if resp.status_code == 200:
                data = resp.json()
                return f"[Cornell LII] {data.get('heading', code_clean)}: {data.get('text', '')[:300]}..."
    except (httpx.TimeoutException, httpx.ConnectError, httpx.HTTPError):
        pass

    # Fallback statutes
    statutes = {
        "26 USC 7201": "Tax Evasion: Willful attempt to evade tax — up to 5 years prison, $250K fine.",
        "UCC 2-725": "Statute of Limitations for UCC contracts: 4 years from breach.",
        "GDPR ART 83": "GDPR Fines: Tier 1 up to EUR 10M/2% revenue; Tier 2 up to EUR 20M/4% revenue.",
        "18 USC 1343": "Wire Fraud: Fine + up to 20 years prison per count.",
    }

    for key, text in statutes.items():
        if key in code_clean or code_clean in key:
            return f"[Mock Statute — API unavailable] {key}: {text}"

    return f"[Mock Statute] {statute_code} not found in local cache. Try: '26 USC 7201', 'UCC 2-725', 'GDPR Art 83'."


# ---------------------------------------------------------------------------
# ReAct agent dùng custom tools
# ---------------------------------------------------------------------------

async def run_agent(question: str) -> str:
    llm = get_llm()
    agent = create_react_agent(
        model=llm,
        tools=[search_legal_database, get_statute_text],
        prompt=(
            "Bạn là luật sư nghiên cứu pháp lý. Sử dụng tools để tra cứu "
            "cơ sở dữ liệu pháp lý và nội dung điều luật. "
            "Luôn dẫn nguồn cụ thể. Tối đa 200 từ."
        ),
    )
    result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]})
    return result["messages"][-1].content


async def main():
    load_dotenv()

    print("=" * 70)
    print("CHALLENGE 3: Custom Tool gọi API thực")
    print("=" * 70)
    print()
    print("[Tools được implement]")
    print("  1. search_legal_database — gọi Case Law API (fallback: mock DB)")
    print("  2. get_statute_text — gọi Cornell LII API (fallback: mock statutes)")
    print()
    print("[Pattern]")
    print("  async with httpx.AsyncClient() as client:")
    print("      resp = await client.get(url, params=params)")
    print("  → Luôn có fallback khi API timeout hoặc unavailable")
    print()

    questions = [
        "What are the penalties for tax evasion under 26 USC 7201?",
        "Find case law about breach of contract and what damages are available.",
    ]

    for q in questions:
        print(f"\nCâu hỏi: {q}")
        print("-" * 70)
        answer = await run_agent(q)
        print(f"Trả lời:\n{answer}")

    print("\n" + "=" * 70)
    print("[Điểm học] httpx.AsyncClient() là async equivalent của requests.")
    print("  - Luôn dùng async with để tự động đóng connection")
    print("  - Set timeout để tránh hang indefinitely")
    print("  - Implement fallback cho production reliability")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
