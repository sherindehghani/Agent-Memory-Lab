#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))

from app.autoresearch.repo import AutoresearchRepo
from app.autoresearch.session import AutoresearchCreate, launch
from app.core.config import get_settings
from app.core.database import new_session
from app.core.logging import configure_logging
from app.models import Run
from app.orchestration.manager import RunManager
from app.orchestration.recovery import bootstrap


def main() -> int:
    parser = argparse.ArgumentParser(description="Run an autoresearch session: the agent edits train.py, keeps "
                                                 "improvements and resets the rest.")
    parser.add_argument("--proposer", choices=["offline", "llm"], default="offline")
    parser.add_argument("--experiments", type=int, default=20)
    parser.add_argument("--minutes", type=float, default=30)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    settings = get_settings()
    configure_logging("WARNING", json_logs=False)
    bootstrap(settings)
    run_id = launch(RunManager(settings), AutoresearchCreate(proposer=args.proposer, max_experiments=args.experiments,
                                                             max_minutes=args.minutes, tag=args.tag, seed=args.seed),
                    background=False)
    repo = AutoresearchRepo(settings.workspace_dir)
    with new_session() as s:
        run = s.get(Run, run_id)
        print(f"session {run_id}: {run.state} ({run.stop_reason or run.error})")
        print(f"branch: {run.initial_configuration.get('branch')}\n")
    print(repo.results_path.read_text() if repo.results_path.exists() else "no results")
    print(f"best train.py ({repo.train_path}):\n")
    print(repo.read_train())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
