"""prompts.py

Small interactive helpers.

Keep them isolated so the rest of the pipeline stays testable.
"""

from __future__ import annotations

import sys


def confirm_or_abort(message: str, token: str = "execute") -> None:
    """Require an explicit confirmation token.

    Args:
        message: Printed prompt for the user.
        token: Required exact token to proceed.

    Raises:
        SystemExit: if token is not provided by the user.
    """

    print("\n" + "=" * 80)
    print(message)
    print(f"Type <{token}> to continue. Anything else aborts.")
    print("=" * 80)
    ans = input("> ").strip()
    if ans != token:
        raise SystemExit("Aborted by user.")


def warn(message: str) -> None:
    print(f"[WARN] {message}", file=sys.stderr)


def info(message: str) -> None:
    print(f"[INFO] {message}")


def error(message: str) -> None:
    print(f"[ERROR] {message}", file=sys.stderr)
