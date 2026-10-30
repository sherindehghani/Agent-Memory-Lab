from __future__ import annotations

import ast
import hashlib
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.autoresearch.templates import HARNESS_CONTRACT

CODE_SCHEMA = {
    "type": "object",
    "properties": {
        "description": {"type": "string"},
        "hypothesis": {"type": "string"},
        "train_py": {"type": "string"},
    },
    "required": ["description", "hypothesis", "train_py"],
    "additionalProperties": False,
}


def source_hash(source: str) -> str:
    normalised = "\n".join(line.rstrip() for line in source.strip().splitlines())
    return hashlib.sha256(normalised.encode()).hexdigest()[:16]


@dataclass
class Proposal:
    description: str
    hypothesis: str
    source: str
    idea: str | None = None
    usage: dict[str, Any] | None = None


@dataclass
class ProposalContext:
    program: str
    current_source: str
    best_f1: float | None
    history: list[dict[str, Any]]
    memories: list[dict[str, Any]]
    tried_hashes: set[str] = field(default_factory=set)
    fix_error: str | None = None
    failed_source: str | None = None


def _ensure_imports(source: str, imports: list[str]) -> str:
    lines = source.splitlines()
    missing = [imp for imp in imports if imp not in lines]
    if not missing:
        return source
    insert_at = 0
    for i, line in enumerate(lines):
        if line.startswith(("import ", "from ")):
            insert_at = i + 1
    return "\n".join(lines[:insert_at] + missing + lines[insert_at:]) + "\n"


def _replace_build_model(source: str, body: str, imports: list[str]) -> str:
    tree = ast.parse(source)
    fn = next((n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "build_model"), None)
    if fn is None:
        return source
    lines = source.splitlines()
    start = (fn.decorator_list[0].lineno if fn.decorator_list else fn.lineno) - 1
    new_fn = ["def build_model(seed):", *[f"    {b}" for b in body.strip().splitlines()]]
    out = "\n".join(lines[:start] + new_fn + lines[fn.end_lineno:]).rstrip() + "\n"
    return _prune_imports(_ensure_imports(out, imports))


def _prune_imports(source: str) -> str:
    tree = ast.parse(source)
    used = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    lines = source.splitlines()
    drop = set()
    for node in tree.body:
        if isinstance(node, ast.ImportFrom) and node.module != "__future__":
            if all((a.asname or a.name) not in used for a in node.names):
                drop.add(node.lineno - 1)
        elif isinstance(node, ast.Import):
            if all((a.asname or a.name).split(".")[0] not in used for a in node.names):
                drop.add(node.lineno - 1)
    return "\n".join(line for i, line in enumerate(lines) if i not in drop).lstrip("\n") + "\n"


def _rewrite(body: str, imports: list[str]) -> Callable[[str], str | None]:
    return lambda src: _replace_build_model(src, body, imports)


def _scale_param(name: str, factor: float, lo: float, hi: float, integer: bool = False) -> Callable[[str], str | None]:
    pattern = re.compile(rf"\b{name}=([0-9]*\.?[0-9]+(?:[eE][-+]?[0-9]+)?)")

    def apply(src: str) -> str | None:
        m = pattern.search(src)
        if not m:
            return None
        value = min(hi, max(lo, float(m.group(1)) * factor))
        text = str(int(round(value))) if integer else f"{value:.4g}"
        return src[: m.start(1)] + text + src[m.end(1):]

    return apply


def _insert_kwarg(call: str, kwarg: str) -> Callable[[str], str | None]:
    def apply(src: str) -> str | None:
        if f"{call}(" not in src or kwarg.split("=")[0] + "=" in src:
            return None
        return src.replace(f"{call}(", f"{call}({kwarg}, ", 1)

    return apply


def _insert_step(step: str, imports: list[str]) -> Callable[[str], str | None]:
    def apply(src: str) -> str | None:
        anchor = "make_pipeline(StandardScaler(), "
        if anchor not in src or step.split("(")[0] in src:
            return None
        return _ensure_imports(src.replace(anchor, anchor + step + ", ", 1), imports)

    return apply


def _remove_text(text: str) -> Callable[[str], str | None]:
    def apply(src: str) -> str | None:
        return _prune_imports(src.replace(text, "", 1)) if text in src else None

    return apply


SCALE = ["from sklearn.pipeline import make_pipeline", "from sklearn.preprocessing import StandardScaler"]


@dataclass(frozen=True)
class Idea:
    name: str
    description: str
    hypothesis: str
    apply: Callable[[str], str | None]


IDEAS: list[Idea] = [
    Idea("logreg", "switch to scaled logistic regression",
         "A linear model on standardised features may separate these classes better than a forest.",
         _rewrite("return make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=5000, random_state=seed))",
                  ["from sklearn.linear_model import LogisticRegression", *SCALE])),
    Idea("svc_rbf", "switch to scaled RBF support vector machine",
         "An RBF SVM can capture mild non-linearity that a linear model misses.",
         _rewrite("return make_pipeline(StandardScaler(), SVC(C=1.0, kernel=\"rbf\", random_state=seed))",
                  ["from sklearn.svm import SVC", *SCALE])),
    Idea("extra_trees", "switch to extra trees (300 trees)",
         "Extra randomisation in split selection may reduce variance compared with a random forest.",
         _rewrite("return ExtraTreesClassifier(n_estimators=300, random_state=seed, n_jobs=1)",
                  ["from sklearn.ensemble import ExtraTreesClassifier"])),
    Idea("hist_gb", "switch to histogram gradient boosting",
         "Boosting may fit decision boundaries more precisely than bagged trees.",
         _rewrite("return HistGradientBoostingClassifier(random_state=seed)",
                  ["from sklearn.ensemble import HistGradientBoostingClassifier"])),
    Idea("c_down", "lower regularisation strength C by sqrt(10)",
         "Stronger regularisation may generalise better on 455 samples.",
         _scale_param("C", 1 / math.sqrt(10), 1e-3, 1e3)),
    Idea("c_up", "raise C by sqrt(10)",
         "Weaker regularisation may let the model fit useful detail.",
         _scale_param("C", math.sqrt(10), 1e-3, 1e3)),
    Idea("balanced_lr", "class_weight=balanced for logistic regression",
         "Re-weighting the minority class may improve recall and therefore F1.",
         _insert_kwarg("LogisticRegression", "class_weight=\"balanced\"")),
    Idea("balanced_svc", "class_weight=balanced for the SVM",
         "Re-weighting the minority class may improve recall and therefore F1.",
         _insert_kwarg("SVC", "class_weight=\"balanced\"")),
    Idea("select_k", "add SelectKBest(k=20) after scaling",
         "Dropping weakly informative features may reduce noise.",
         _insert_step("SelectKBest(f_classif, k=20)",
                      ["from sklearn.feature_selection import SelectKBest, f_classif"])),
    Idea("poly", "add degree-2 interaction features after scaling",
         "Pairwise interactions may make the classes more linearly separable.",
         _insert_step("PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)",
                      ["from sklearn.preprocessing import PolynomialFeatures"])),
    Idea("drop_select_k", "remove SelectKBest (simplify)",
         "The feature selector may not be pulling its weight; simpler code at equal F1 is a win.",
         _remove_text("SelectKBest(f_classif, k=20), ")),
    Idea("drop_poly", "remove interaction features (simplify)",
         "Interaction features may be unnecessary; simpler code at equal F1 is a win.",
         _remove_text("PolynomialFeatures(degree=2, interaction_only=True, include_bias=False), ")),
    Idea("more_trees", "double n_estimators",
         "More trees reduce ensemble variance.",
         _scale_param("n_estimators", 2, 10, 1000, integer=True)),
    Idea("fewer_trees", "halve n_estimators",
         "Fewer trees may score the same with less memory and time.",
         _scale_param("n_estimators", 0.5, 10, 1000, integer=True)),
    Idea("soft_vote", "soft-voting ensemble of LR, SVM and random forest",
         "Averaging diverse models may beat each one alone.",
         _rewrite("""lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, random_state=seed))
svc = make_pipeline(StandardScaler(), SVC(probability=True, random_state=seed))
rf = RandomForestClassifier(n_estimators=200, random_state=seed, n_jobs=1)
return VotingClassifier([("lr", lr), ("svc", svc), ("rf", rf)], voting="soft")""",
                  ["from sklearn.ensemble import RandomForestClassifier, VotingClassifier",
                   "from sklearn.linear_model import LogisticRegression", "from sklearn.svm import SVC", *SCALE])),
]


class OfflineProposer:
    name = "offline"

    def propose(self, ctx: ProposalContext) -> Proposal | None:
        for idea in IDEAS:
            try:
                new = idea.apply(ctx.current_source)
            except SyntaxError:
                continue
            if not new or source_hash(new) == source_hash(ctx.current_source) or source_hash(new) in ctx.tried_hashes:
                continue
            return Proposal(idea.description, idea.hypothesis, new, idea=idea.name)
        return None

    def fix(self, ctx: ProposalContext) -> Proposal | None:
        return None


class LLMProposer:
    name = "llm"

    def __init__(self, client: Any):
        self.client = client

    def _ask(self, user: str) -> Proposal:
        data, usage = self.client.complete_json(HARNESS_CONTRACT, user, schema=CODE_SCHEMA, max_tokens=16000,
                                                effort="medium")
        return Proposal(data["description"].strip()[:120], data["hypothesis"].strip(), data["train_py"],
                        idea="llm", usage=usage)

    def propose(self, ctx: ProposalContext) -> Proposal | None:
        user = "\n\n".join([
            "## program.md\n" + ctx.program,
            "## current train.py (last kept commit)\n```python\n" + ctx.current_source + "\n```",
            f"## best val_f1 so far\n{ctx.best_f1}",
            "## results history (newest last)\n" + "\n".join(
                f"{r['commit'][:7]}\t{r['val_f1']:.6f}\t{r['status']}\t{r['description']}" for r in ctx.history[-40:]),
            "## memories from earlier experiments and sessions\n" + json.dumps(ctx.memories[:20], default=str),
            "Propose the next experiment. It must differ from every configuration in the history.",
        ])
        return self._ask(user)

    def fix(self, ctx: ProposalContext) -> Proposal | None:
        user = "\n\n".join([
            "The previous train.py failed. Fix the bug while keeping the idea, or return a minimal safe change.",
            "## failing train.py\n```python\n" + (ctx.failed_source or "") + "\n```",
            "## error output (tail)\n```\n" + (ctx.fix_error or "")[-4000:] + "\n```",
        ])
        return self._ask(user)
