"""
Shared token-counting utility.

Uses tiktoken (cl100k_base) when available; falls back to len(text)//4.
Import and use count_tokens() everywhere instead of inline estimates.
"""

import logging

logger = logging.getLogger("mcp_backend")

_TIKTOKEN_ENC = None


def _get_encoder():
    global _TIKTOKEN_ENC
    if _TIKTOKEN_ENC is None:
        try:
            import tiktoken
            _TIKTOKEN_ENC = tiktoken.get_encoding("cl100k_base")
        except Exception as e:
            logger.warning(f"[TOKEN COUNTER] tiktoken unavailable, using char/4 fallback: {e}")
    return _TIKTOKEN_ENC


def count_tokens(text: str) -> int:
    """Return approximate token count for *text*."""
    if not text:
        return 0
    try:
        enc = _get_encoder()
        if enc:
            return len(enc.encode(text))
    except Exception:
        pass
    return len(text) // 4
