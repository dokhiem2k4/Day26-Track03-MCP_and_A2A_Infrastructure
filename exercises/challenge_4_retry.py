"""Challenge 4: Retry Logic với Exponential Backoff

Khi một agent hoặc tool gặp lỗi tạm thời (network timeout, rate limit,
service unavailable), hệ thống tự động thử lại thay vì fail ngay.

Pattern: exponential backoff với jitter
  Attempt 1: thử ngay
  Attempt 2: chờ 1s  (+/- jitter)
  Attempt 3: chờ 2s  (+/- jitter)
  Attempt 4: chờ 4s  (+/- jitter)
  ...sau max_attempts: raise exception

Chạy:
    uv run python exercises/challenge_4_retry.py
"""

import asyncio
import os
import random
import sys
import time
from functools import wraps
from typing import TypeVar, Callable, Any

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dotenv import load_dotenv
from langchain_core.messages import HumanMessage
from langchain_core.tools import tool

from common.llm import get_llm


T = TypeVar("T")


# ---------------------------------------------------------------------------
# Retry decorator với exponential backoff
# ---------------------------------------------------------------------------

class RetryExhaustedError(Exception):
    """Raised khi đã hết số lần retry mà vẫn fail."""
    pass


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    jitter: bool = True,
    retryable_exceptions: tuple = (Exception,),
):
    """Decorator: tự động retry async function với exponential backoff.

    Args:
        max_attempts: Số lần thử tối đa (bao gồm lần đầu tiên)
        base_delay: Thời gian chờ cơ bản (giây) giữa các lần thử
        max_delay: Thời gian chờ tối đa (giây)
        jitter: Thêm ngẫu nhiên vào delay để tránh thundering herd
        retryable_exceptions: Chỉ retry khi gặp các exception này
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None

            for attempt in range(1, max_attempts + 1):
                try:
                    return await func(*args, **kwargs)

                except retryable_exceptions as exc:
                    last_exception = exc

                    if attempt == max_attempts:
                        raise RetryExhaustedError(
                            f"{func.__name__} failed after {max_attempts} attempts. "
                            f"Last error: {exc}"
                        ) from exc

                    # Tính delay: 2^(attempt-1) * base_delay
                    delay = min(base_delay * (2 ** (attempt - 1)), max_delay)
                    if jitter:
                        delay *= (0.5 + random.random() * 0.5)  # 50-100% của delay

                    print(
                        f"    [retry] {func.__name__} attempt {attempt}/{max_attempts} "
                        f"failed: {exc}. Retrying in {delay:.1f}s..."
                    )
                    await asyncio.sleep(delay)

            raise RetryExhaustedError(f"Unreachable") from last_exception

        return wrapper
    return decorator


# ---------------------------------------------------------------------------
# Simulated flaky service (để demo retry hoạt động)
# ---------------------------------------------------------------------------

_call_count = 0


async def _flaky_llm_call(prompt: str, fail_first_n: int = 2) -> str:
    """Giả lập LLM API bị lỗi trong N lần đầu."""
    global _call_count
    _call_count += 1

    if _call_count <= fail_first_n:
        raise ConnectionError(f"Simulated network error (attempt {_call_count})")

    # Lần gọi thực sự
    llm = get_llm()
    response = llm.invoke([HumanMessage(content=prompt)])
    return response.content


# ---------------------------------------------------------------------------
# Wrapped version với retry
# ---------------------------------------------------------------------------

@with_retry(max_attempts=4, base_delay=0.5, jitter=True)
async def resilient_llm_call(prompt: str) -> str:
    """LLM call với automatic retry."""
    return await _flaky_llm_call(prompt)


# ---------------------------------------------------------------------------
# Tool-level retry
# ---------------------------------------------------------------------------

_tool_call_count = 0


@tool
async def search_with_retry(query: str) -> str:
    """Tìm kiếm pháp lý với retry tự động khi gặp lỗi mạng.

    Args:
        query: Câu truy vấn pháp lý
    """
    global _tool_call_count
    _tool_call_count += 1

    # Giả lập tool bị lỗi lần đầu
    if _tool_call_count == 1:
        raise TimeoutError("Simulated API timeout on first call")

    return (
        f"[Legal DB] Kết quả cho '{query}': "
        "Tìm thấy 3 án lệ liên quan. "
        "Vụ quan trọng nhất: Hadley v. Baxendale (1854) về consequential damages."
    )


async def call_tool_with_retry(query: str, max_attempts: int = 3) -> str:
    """Gọi tool với retry loop thủ công (không dùng decorator)."""
    for attempt in range(1, max_attempts + 1):
        try:
            return search_with_retry.invoke({"query": query})
        except (TimeoutError, ConnectionError) as exc:
            if attempt == max_attempts:
                return f"Tool failed after {max_attempts} attempts: {exc}"
            delay = 0.5 * (2 ** (attempt - 1))
            print(f"    [tool retry] Attempt {attempt} failed: {exc}. Retrying in {delay:.1f}s...")
            await asyncio.sleep(delay)
    return "Unreachable"


# ---------------------------------------------------------------------------
# Agent-level retry với graceful degradation
# ---------------------------------------------------------------------------

async def law_agent_with_retry(question: str, max_attempts: int = 3) -> str:
    """Law agent với retry + graceful degradation nếu hết lần thử."""
    for attempt in range(1, max_attempts + 1):
        try:
            llm = get_llm()
            response = await llm.ainvoke([HumanMessage(content=(
                f"Bạn là luật sư. Trả lời ngắn gọn (<100 từ):\n{question}"
            ))])
            return response.content

        except Exception as exc:
            if attempt == max_attempts:
                # Graceful degradation: trả về disclaimer thay vì crash
                return (
                    f"[Degraded Response] Không thể kết nối LLM sau {max_attempts} lần thử "
                    f"(lỗi: {exc}). Vui lòng thử lại sau hoặc liên hệ luật sư trực tiếp."
                )

            delay = 1.0 * (2 ** (attempt - 1))
            print(f"    [agent retry] Attempt {attempt} failed: {exc}. Retrying in {delay:.1f}s...")
            await asyncio.sleep(delay)

    return "Unreachable"


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

async def main():
    load_dotenv()

    print("=" * 70)
    print("CHALLENGE 4: Retry Logic với Exponential Backoff")
    print("=" * 70)
    print()

    # --- Demo 1: Decorator-based retry ---
    print("[Demo 1] @with_retry decorator trên LLM call")
    print("  Giả lập: fail 2 lần đầu, thành công lần 3")
    print("-" * 50)
    global _call_count
    _call_count = 0  # reset counter

    start = time.time()
    try:
        result = await resilient_llm_call(
            "Tóm tắt trong 1 câu: breach of contract nghĩa là gì?"
        )
        elapsed = time.time() - start
        print(f"  Thanh cong sau {elapsed:.1f}s:\n  {result[:150]}...")
    except RetryExhaustedError as e:
        print(f"  Het retry: {e}")

    print()

    # --- Demo 2: Manual retry cho tool calls ---
    print("[Demo 2] Manual retry cho tool call")
    print("  Giả lập: tool timeout lần đầu, thành công lần 2")
    print("-" * 50)
    global _tool_call_count
    _tool_call_count = 0  # reset counter

    result = await call_tool_with_retry("breach of contract damages")
    print(f"  Ket qua: {result}")

    print()

    # --- Demo 3: Agent với graceful degradation ---
    print("[Demo 3] Law agent với graceful degradation")
    print("  Gọi thực tế LLM (không giả lập lỗi)")
    print("-" * 50)

    question = "Điều kiện để hợp đồng có hiệu lực pháp lý là gì?"
    answer = await law_agent_with_retry(question)
    print(f"  Q: {question}")
    print(f"  A: {answer[:200]}...")

    print()
    print("=" * 70)
    print("[Tổng kết các pattern]")
    print()
    print("  1. @with_retry decorator — áp dụng cho bất kỳ async function nào")
    print("     + Tái sử dụng được, khai báo ở function level")
    print("     - Cần import decorator vào mỗi file")
    print()
    print("  2. Manual retry loop — linh hoạt hơn, dễ đọc")
    print("     + Có thể thêm logic phức tạp giữa các lần thử")
    print("     - Code dài hơn, dễ quên handle edge cases")
    print()
    print("  3. Graceful degradation — không crash, trả về thông báo lỗi")
    print("     + User experience tốt hơn")
    print("     - Cần cẩn thận không che giấu lỗi thực sự")
    print()
    print("  Exponential backoff: delay = base * 2^(attempt-1)")
    print("  Jitter: nhân thêm random 50-100% để tránh thundering herd")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
