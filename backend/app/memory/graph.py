from __future__ import annotations

from collections import deque
from typing import Any

import networkx as nx
from sqlalchemy import exists, select
from sqlalchemy.orm import Session

from app.models import (
    Agent,
    AgentAction,
    Experiment,
    ExperimentMetric,
    Finding,
    GraphRelationship,
    MemoryRecord,
    Run,
)

NODE_TYPES = ("experiment", "agent", "model", "config", "finding", "memory", "commit", "run")
RELATIONS = (
    "PROPOSED", "EXECUTED", "EVALUATED", "USES", "TRAINS", "PRODUCED", "SUPPORTED_BY", "DERIVED_FROM",
    "RECORDED_IN", "REFERS_TO", "INFLUENCED", "PART_OF", "COMPARED_AGAINST",
)
EVIDENCE_TYPES = NODE_TYPES + ("action",)


class GraphEvidenceError(ValueError):
    pass


def split_node(node_id: str) -> tuple[str, str]:
    node_type, _, key = node_id.partition(":")
    if node_type not in EVIDENCE_TYPES or not key:
        raise GraphEvidenceError(f"malformed node reference {node_id!r}")
    return node_type, key


def node_exists(session: Session, node_id: str) -> bool:
    node_type, key = split_node(node_id)
    direct = {"experiment": Experiment, "run": Run, "agent": Agent, "finding": Finding, "memory": MemoryRecord,
              "action": AgentAction}
    if node_type in direct:
        return session.get(direct[node_type], key) is not None
    column = {"model": Experiment.model_type, "config": Experiment.config_hash,
              "commit": Experiment.git_commit}[node_type]
    return bool(session.scalar(select(exists().where(column == key))))


def add_relationship(session: Session, namespace: str, source: str, relation: str, target: str,
                     evidence_ref: str) -> GraphRelationship:
    if relation not in RELATIONS:
        raise GraphEvidenceError(f"unknown relation {relation!r}")
    for ref, label in ((source, "source"), (target, "target"), (evidence_ref, "evidence")):
        if label != "evidence" and split_node(ref)[0] not in NODE_TYPES:
            raise GraphEvidenceError(f"{label} {ref!r} is not a graph node type")
        if not node_exists(session, ref):
            raise GraphEvidenceError(f"{label} {ref!r} does not reference an existing record")
    existing = session.scalar(select(GraphRelationship).where(
        GraphRelationship.source_id == source, GraphRelationship.relation == relation,
        GraphRelationship.target_id == target))
    if existing is not None:
        return existing
    rel = GraphRelationship(namespace=namespace, source_id=source, relation=relation, target_id=target,
                            evidence_ref=evidence_ref)
    session.add(rel)
    session.flush()
    return rel


def _node_attributes(session: Session, node_ids: set[str]) -> dict[str, dict[str, Any]]:
    by_type: dict[str, set[str]] = {}
    for nid in node_ids:
        try:
            t, k = split_node(nid)
        except GraphEvidenceError:
            continue
        by_type.setdefault(t, set()).add(k)
    attrs: dict[str, dict[str, Any]] = {}

    if keys := by_type.get("experiment"):
        exps = session.scalars(select(Experiment).where(Experiment.id.in_(keys))).all()
        f1 = dict(session.execute(select(ExperimentMetric.experiment_id, ExperimentMetric.value).where(
            ExperimentMetric.experiment_id.in_(keys), ExperimentMetric.name == "f1",
            ExperimentMetric.split == "validation")).all())
        for e in exps:
            attrs[f"experiment:{e.id}"] = {
                "type": "experiment", "label": f"{e.id} ({e.decision or e.status})", "decision": e.decision,
                "status": e.status, "model_type": e.model_type, "configuration": e.configuration,
                "hypothesis": e.hypothesis, "f1": f1.get(e.id), "run_id": e.run_id, "sequence": e.sequence,
                "hypothesis_outcome": e.hypothesis_outcome, "git_commit": e.git_commit,
            }
    if keys := by_type.get("finding"):
        for f in session.scalars(select(Finding).where(Finding.id.in_(keys))):
            attrs[f"finding:{f.id}"] = {"type": "finding", "label": f.finding_type, "statement": f.statement,
                                        "verification_status": f.verification_status, "evidence": f.evidence}
    if keys := by_type.get("memory"):
        for m in session.scalars(select(MemoryRecord).where(MemoryRecord.id.in_(keys))):
            attrs[f"memory:{m.id}"] = {"type": "memory", "label": m.title, "category": m.category, "kind": m.kind,
                                       "verification_status": m.verification_status, "content": m.content}
    if keys := by_type.get("agent"):
        for a in session.scalars(select(Agent).where(Agent.id.in_(keys))):
            attrs[f"agent:{a.id}"] = {"type": "agent", "label": a.name, "role": a.role}
    if keys := by_type.get("run"):
        for r in session.scalars(select(Run).where(Run.id.in_(keys))):
            attrs[f"run:{r.id}"] = {"type": "run", "label": r.id, "state": r.state, "objective": r.objective,
                                    "memory_enabled": r.memory_enabled}
    if keys := by_type.get("config"):
        for e in session.scalars(select(Experiment).where(Experiment.config_hash.in_(keys))):
            attrs.setdefault(f"config:{e.config_hash}", {
                "type": "config", "label": _config_label(e.model_type, e.configuration),
                "model_type": e.model_type, "configuration": e.configuration, "config_hash": e.config_hash})
    if keys := by_type.get("model"):
        present = set(session.scalars(select(Experiment.model_type).where(Experiment.model_type.in_(keys))))
        for k in present:
            attrs[f"model:{k}"] = {"type": "model", "label": k}
    if keys := by_type.get("commit"):
        present = set(session.scalars(select(Experiment.git_commit).where(Experiment.git_commit.in_(keys))))
        for k in present:
            attrs[f"commit:{k}"] = {"type": "commit", "label": k[:10], "sha": k}
    return attrs


def _config_label(model_type: str, configuration: dict[str, Any]) -> str:
    abbrev = {"random_forest": "RF", "logistic_regression": "LR"}.get(model_type, model_type)
    return f"{abbrev} " + ", ".join(f"{k}={v}" for k, v in configuration.items())


def build_graph(session: Session, namespace: str = "main") -> tuple[nx.MultiDiGraph, list[dict[str, Any]]]:
    rels = session.scalars(select(GraphRelationship).where(GraphRelationship.namespace == namespace)).all()
    node_ids = {r.source_id for r in rels} | {r.target_id for r in rels}
    attrs = _node_attributes(session, node_ids)
    graph = nx.MultiDiGraph()
    dangling: list[dict[str, Any]] = []
    for r in rels:
        missing = [n for n in (r.source_id, r.target_id) if n not in attrs]
        if missing:
            dangling.append({"relationship_id": r.id, "source": r.source_id, "relation": r.relation,
                             "target": r.target_id, "missing": missing})
            continue
        for n in (r.source_id, r.target_id):
            if n not in graph:
                graph.add_node(n, **attrs[n])
        graph.add_edge(r.source_id, r.target_id, key=r.relation, relation=r.relation,
                       evidence_ref=r.evidence_ref, created_at=r.created_at.isoformat(), id=r.id)
    return graph, dangling


def _in_edges(graph: nx.MultiDiGraph, node: str, relation: str) -> list[str]:
    return [u for u, _, k in graph.in_edges(node, keys=True) if k == relation] if node in graph else []


def _out_edges(graph: nx.MultiDiGraph, node: str, relation: str) -> list[str]:
    return [v for _, v, k in graph.out_edges(node, keys=True) if k == relation] if node in graph else []


def experiments_in_run(graph: nx.MultiDiGraph, run_id: str) -> list[str]:
    nodes = _in_edges(graph, f"run:{run_id}", "PART_OF")
    return sorted(nodes, key=lambda n: graph.nodes[n].get("sequence", 0))


def derived_from(graph: nx.MultiDiGraph, experiment_id: str) -> list[str]:
    start = f"experiment:{experiment_id}"
    seen, queue, out = {start}, deque([start]), []
    while queue:
        node = queue.popleft()
        for child in _in_edges(graph, node, "DERIVED_FROM"):
            if child not in seen:
                seen.add(child)
                out.append(child)
                queue.append(child)
    return out


def findings_from_rejected(graph: nx.MultiDiGraph) -> list[dict[str, Any]]:
    out = []
    for node, data in graph.nodes(data=True):
        if data.get("type") == "experiment" and data.get("decision") == "rejected":
            for f in _out_edges(graph, node, "PRODUCED"):
                out.append({"experiment": node, "finding": f, **graph.nodes[f]})
    return out


def influences(graph: nx.MultiDiGraph, experiment_id: str) -> list[dict[str, Any]]:
    node = f"experiment:{experiment_id}"
    return [{"memory": m, **graph.nodes[m]} for m in _in_edges(graph, node, "INFLUENCED")]


def evidence_path(graph: nx.MultiDiGraph, finding_id: str) -> dict[str, Any]:
    start = f"finding:{finding_id}"
    if start not in graph:
        return {"finding": start, "found": False, "paths": [], "nodes": {}}
    paths: list[list[dict[str, str]]] = []
    nodes = {start: graph.nodes[start]}
    for exp in _out_edges(graph, start, "SUPPORTED_BY"):
        nodes[exp] = graph.nodes[exp]
        base = [{"source": start, "relation": "SUPPORTED_BY", "target": exp,
                 "evidence_ref": graph.edges[start, exp, "SUPPORTED_BY"]["evidence_ref"]}]
        extended = False
        for relation in ("USES", "RECORDED_IN", "TRAINS", "COMPARED_AGAINST"):
            for target in _out_edges(graph, exp, relation):
                nodes[target] = graph.nodes[target]
                paths.append(base + [{"source": exp, "relation": relation, "target": target,
                                      "evidence_ref": graph.edges[exp, target, relation]["evidence_ref"]}])
                extended = True
        if not extended:
            paths.append(base)
    return {"finding": start, "found": True, "paths": paths, "nodes": nodes}


def neighbourhood(graph: nx.MultiDiGraph, centre: list[str], radius: int = 1, limit: int = 150) -> nx.MultiDiGraph:
    undirected = graph.to_undirected(as_view=True)
    keep: list[str] = []
    for c in centre:
        if c in graph:
            for n in nx.single_source_shortest_path_length(undirected, c, cutoff=radius):
                if n not in keep:
                    keep.append(n)
    return graph.subgraph(keep[:limit]).copy()


def to_flow(graph: nx.MultiDiGraph, dangling: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {
        "nodes": [{"id": n, **data} for n, data in graph.nodes(data=True)],
        "edges": [{"id": f"{u}|{k}|{v}", "source": u, "target": v, "relation": k,
                   "evidence_ref": d["evidence_ref"], "created_at": d["created_at"]}
                  for u, v, k, d in graph.edges(keys=True, data=True)],
        "dangling": dangling or [],
    }
