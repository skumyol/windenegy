#!/usr/bin/env python3
"""Development server runner for API and dashboard.

Usage:
    python scripts/run_dev.py api      # Start API server
    python scripts/run_dev.py dashboard # Start dashboard
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


API_PORT = 8765
DASHBOARD_PORT = 8766


def run_api() -> int:
    """Start the FastAPI server."""
    print(f"Starting API server on http://localhost:{API_PORT}")
    print(f"API docs: http://localhost:{API_PORT}/docs")
    return subprocess.run(
        ["uvicorn", "windenegy.interface.api:app", "--host", "0.0.0.0", "--port", str(API_PORT), "--reload"],
        cwd=str(Path(__file__).parent.parent),
        check=False,
    ).returncode


def run_dashboard() -> int:
    """Start the Streamlit dashboard."""
    print(f"Starting dashboard on http://localhost:{DASHBOARD_PORT}")
    return subprocess.run(
        ["streamlit", "run", "src/windenegy/interface/dashboard.py", f"--server.port={DASHBOARD_PORT}"],
        cwd=str(Path(__file__).parent.parent),
        check=False,
    ).returncode


def main() -> int:
    """Main entry point."""
    parser = argparse.ArgumentParser(description="Run development servers")
    parser.add_argument(
        "service",
        choices=["api", "dashboard"],
        help="Which service to start",
    )
    args = parser.parse_args()

    if args.service == "api":
        return run_api()
    if args.service == "dashboard":
        return run_dashboard()
    return 1


if __name__ == "__main__":
    sys.exit(main())
