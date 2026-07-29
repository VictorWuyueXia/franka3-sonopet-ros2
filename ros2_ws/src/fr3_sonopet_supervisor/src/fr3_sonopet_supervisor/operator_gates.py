from __future__ import annotations


def is_execute_token_valid(token: str, expected: str = "E") -> bool:
    return token.strip() == expected

