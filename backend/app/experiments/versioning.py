"""Git-backed experiment versioning in an isolated workspace repository.

Each experiment gets a commit whose *parent is the commit of the experiment it
was derived from* (built with ``git commit-tree``), so the Git DAG mirrors
experiment derivation and ``git diff <parent> <child>`` shows exactly the
configuration change. Commits are referenced by ``refs/experiments/<id>``.

Only small JSON config artifacts are committed. Metrics live in the database;
the accepted baseline is a database pointer, never inferred from Git.
All commands are argument lists (no shell) and paths are confined to the workspace.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from pathlib import Path
from typing import Any

ID_PATTERN = re.compile(r"^[a-z]+_[0-9a-f]{12}$")
SHA_PATTERN = re.compile(r"^[0-9a-f]{7,64}$")
WORKSPACE_GITIGNORE = "*.pkl\n*.joblib\n*.db\n*.sqlite*\n*.csv\n*.parquet\n*.npy\n.env*\n*.key\n"
WORKSPACE_README = (
    "# Experiment workspace\n\nManaged by AI Agent Memory Lab. Each experiment is a commit under\n"
    "`refs/experiments/<id>` whose parent is the experiment it was derived from.\n\n"
    "    git log --graph --oneline refs/experiments/\n"
)


class GitError(RuntimeError):
    pass


class GitWorkspace:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self.repo = self.root / "experiments"
        self.artifacts = self.root / "artifacts"

    # -- helpers ---------------------------------------------------------
    def _git(self, *args: str, input: str | None = None) -> str:
        env = {
            **os.environ,
            "GIT_AUTHOR_NAME": "AI Agent Memory Lab",
            "GIT_AUTHOR_EMAIL": "lab@localhost",
            "GIT_COMMITTER_NAME": "AI Agent Memory Lab",
            "GIT_COMMITTER_EMAIL": "lab@localhost",
            "GIT_TERMINAL_PROMPT": "0",
            "GIT_CONFIG_NOSYSTEM": "1",
        }
        try:
            result = subprocess.run(
                ["git", "-C", str(self.repo), *args], input=input, capture_output=True, text=True,
                timeout=30, env=env, check=False,
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise GitError(f"git {args[0]} failed: {exc}") from exc
        if result.returncode != 0:
            raise GitError(f"git {' '.join(args[:2])} failed: {result.stderr.strip()}")
        return result.stdout.strip()

    def safe_path(self, *parts: str) -> Path:
        path = self.root.joinpath(*parts).resolve()
        if path != self.root and self.root not in path.parents:
            raise GitError(f"path escapes workspace: {path}")
        return path

    @staticmethod
    def _check_id(experiment_id: str) -> None:
        if not ID_PATTERN.match(experiment_id):
            raise GitError(f"invalid experiment id {experiment_id!r}")

    # -- lifecycle -------------------------------------------------------
    def init(self) -> str:
        """Create the workspace repo (idempotent). Returns the root commit."""
        self.repo.mkdir(parents=True, exist_ok=True)
        self.artifacts.mkdir(parents=True, exist_ok=True)
        if not (self.repo / ".git").exists():
            self._git("init", "-q", "-b", "main")
            (self.repo / ".gitignore").write_text(WORKSPACE_GITIGNORE)
            (self.repo / "README.md").write_text(WORKSPACE_README)
            self._git("add", ".gitignore", "README.md")
            self._git("commit", "-q", "-m", "Initialize experiment workspace")
        return self.root_commit()

    def root_commit(self) -> str:
        return self._git("rev-parse", "refs/heads/main")

    def commit_experiment(self, experiment_id: str, artifact: dict[str, Any], parent_commit: str | None,
                          message: str) -> str:
        """Commit ``config.json`` for an experiment on top of ``parent_commit``."""
        self._check_id(experiment_id)
        parent = parent_commit if parent_commit and self.commit_exists(parent_commit) else self.root_commit()
        content = json.dumps(artifact, indent=2, sort_keys=True) + "\n"
        blob = self._git("hash-object", "-w", "--stdin", input=content)
        entries = {
            line.split("\t", 1)[1]: line
            for line in self._git("ls-tree", parent).splitlines()
            if "\t" in line
        }
        entries["config.json"] = f"100644 blob {blob}\tconfig.json"
        tree = self._git("mktree", input="\n".join(entries[k] for k in sorted(entries)) + "\n")
        commit = self._git("commit-tree", tree, "-p", parent, "-m", message[:500])
        self._git("update-ref", f"refs/experiments/{experiment_id}", commit)
        self.save_artifact(experiment_id, "config.json", artifact)
        return commit

    # -- queries ---------------------------------------------------------
    def commit_exists(self, sha: str | None) -> bool:
        if not sha or not SHA_PATTERN.match(sha):
            return False
        try:
            self._git("cat-file", "-e", f"{sha}^{{commit}}")
            return True
        except GitError:
            return False

    def history(self, sha: str, limit: int = 20) -> list[dict[str, Any]]:
        if not self.commit_exists(sha):
            return []
        out = self._git("log", f"-n{limit}", "--format=%H%x1f%P%x1f%s%x1f%cI", sha)
        rows = []
        for line in out.splitlines():
            commit, parents, subject, date = line.split("\x1f")
            rows.append({"commit": commit, "parents": parents.split(), "message": subject, "date": date})
        return rows

    def show_config(self, sha: str) -> dict[str, Any] | None:
        if not self.commit_exists(sha):
            return None
        try:
            return json.loads(self._git("show", f"{sha}:config.json"))
        except (GitError, json.JSONDecodeError):
            return None

    def diff(self, parent: str, child: str) -> str:
        if not (self.commit_exists(parent) and self.commit_exists(child)):
            return ""
        return self._git("diff", parent, child, "--", "config.json")

    def save_artifact(self, experiment_id: str, name: str, data: dict[str, Any]) -> Path:
        self._check_id(experiment_id)
        if "/" in name or name.startswith("."):
            raise GitError(f"invalid artifact name {name!r}")
        path = self.safe_path("artifacts", experiment_id, name)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data, indent=2, sort_keys=True, default=str) + "\n")
        return path
