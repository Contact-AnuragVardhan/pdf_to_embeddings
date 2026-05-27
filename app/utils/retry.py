from __future__ import annotations

from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential_jitter


transient_retry = retry(
    retry=retry_if_exception_type(Exception),
    wait=wait_exponential_jitter(initial=1, max=30),
    stop=stop_after_attempt(6),
    reraise=True,
)
