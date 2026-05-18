from __future__ import annotations


def is_execute_token_valid(token: str, expected: str = "EXECUTE") -> bool:
    return token.strip() == expected

