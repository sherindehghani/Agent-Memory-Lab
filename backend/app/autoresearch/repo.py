from __future__ import annotations

import re
from pathlib import Path

from app.autoresearch.templates import BASELINE_TRAIN, PROGRAM_TEMPLATE
from app.experiments.versioning import GitError, GitWorkspace

RESULTS_HEADER = "commit\tval_f1\tmemory_mb\tstatus\tdescription\n"
TAG_PATTERN = re.compile(r"^[a-z0-9][a-z0-9-]{0,39}$")


class AutoresearchRepo(GitWorkspace):
    def __init__(self, root: Path):
        super().__init__(root)
        self.repo = self.root / "autoresearch"
        self.train_path = self.repo / "train.py"
        self.program_path = self.repo / "program.md"
        self.results_path = self.repo / "results.tsv"
        self.logs = self.repo / "logs"

    def init(self) -> str:
        self.repo.mkdir(parents=True, exist_ok=True)
        if not (self.repo / ".git").exists():
            self._git("init", "-q", "-b", "master")
            (self.repo / ".gitignore").write_text("results.tsv\nprogram.md\nlogs/\n__pycache__/\n")
            self.train_path.write_text(BASELINE_TRAIN)
            self._git("add", ".gitignore", "train.py")
            self._git("commit", "-q", "-m", "baseline train.py")
        if not self.program_path.exists():
            self.program_path.write_text(PROGRAM_TEMPLATE)
        return self.head()

    def head(self) -> str:
        return self._git("rev-parse", "HEAD")

    def branch(self) -> str:
        return self._git("rev-parse", "--abbrev-ref", "HEAD")

    def start_branch(self, tag: str, start: str = "master") -> str:
        if not TAG_PATTERN.match(tag):
            raise GitError(f"invalid tag {tag!r}")
        self._git("reset", "-q", "--hard")
        existing = set(self._git("branch", "--format=%(refname:short)").split())
        name, n = f"autoresearch/{tag}", 2
        while name in existing:
            name, n = f"autoresearch/{tag}-{n}", n + 1
        self._git("checkout", "-q", "-B", name, start)
        return name

    def read_train(self, rev: str | None = None) -> str:
        if rev is None:
            return self.train_path.read_text()
        return self._git_raw("show", f"{rev}:train.py")

    def commit_train(self, source: str, message: str) -> str:
        self.train_path.write_text(source)
        self._git("add", "train.py")
        self._git("commit", "-q", "--allow-empty", "-m", message[:200])
        return self.head()

    def keep_attempt(self, experiment_id: str, sha: str) -> None:
        self._check_id(experiment_id)
        self._git("update-ref", f"refs/autoresearch/attempts/{experiment_id}", sha)

    def reset_to(self, sha: str) -> None:
        self._git("reset", "-q", "--hard", sha)

    def diff(self, a: str, b: str) -> str:
        if not (self.commit_exists(a) and self.commit_exists(b)):
            return ""
        return self._git_raw("diff", a, b, "--", "train.py")

    def append_result(self, commit: str, val_f1: float, memory_mb: float, status: str, description: str) -> None:
        if not self.results_path.exists():
            self.results_path.write_text(RESULTS_HEADER)
        clean = " ".join(description.split()).replace("\t", " ")
        with open(self.results_path, "a") as f:
            f.write(f"{commit[:7]}\t{val_f1:.6f}\t{memory_mb:.3f}\t{status}\t{clean}\n")

    def read_program(self) -> str:
        return self.program_path.read_text() if self.program_path.exists() else PROGRAM_TEMPLATE

    def write_program(self, content: str) -> None:
        self.repo.mkdir(parents=True, exist_ok=True)
        self.program_path.write_text(content)

    def _git_raw(self, *args: str) -> str:
        import os
        import subprocess

        result = subprocess.run(["git", "-C", str(self.repo), *args], capture_output=True, text=True, timeout=30,
                                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"}, check=False)
        if result.returncode != 0:
            raise GitError(f"git {args[0]} failed: {result.stderr.strip()}")
        return result.stdout
