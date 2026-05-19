from __future__ import annotations

import subprocess


def notify(title: str, message: str) -> None:
    script = f'display notification "{message}" with title "{title}"'
    try:
        subprocess.run(["osascript", script], capture_output=True, timeout=10)
    except Exception:
        pass  # Notification is best-effort
