from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query
from sqlalchemy import func, or_, select
from sqlalchemy.orm import Session

from app.api.common import MAX_PAGE, not_found, paginate
from app.core.database import get_db
from app.models import Agent, AgentAction, AgentMessage

router = APIRouter(prefix="/api/agents", tags=["agents"])


@router.get("")
def list_agents(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    counts = dict(db.execute(select(AgentAction.agent_id, func.count()).group_by(AgentAction.agent_id)).all())
    return [{"id": a.id, "name": a.name, "role": a.role, "description": a.description, "tools": a.tools,
             "action_count": counts.get(a.id, 0)} for a in db.scalars(select(Agent).order_by(Agent.id))]


@router.get("/{agent_id}/activity")
def agent_activity(agent_id: str, run_id: str | None = None, limit: int = Query(50, ge=1, le=MAX_PAGE),
                   offset: int = Query(0, ge=0), db: Session = Depends(get_db)) -> dict[str, Any]:
    """Actions (with retrieved memories) and messages sent/received by an agent."""
    if db.get(Agent, agent_id) is None:
        raise not_found("agent", agent_id)
    stmt = select(AgentAction).where(AgentAction.agent_id == agent_id)
    msg_stmt = select(AgentMessage).where(or_(AgentMessage.sender == agent_id,
                                              AgentMessage.recipient.in_([agent_id, "broadcast"])))
    if run_id:
        stmt = stmt.where(AgentAction.run_id == run_id)
        msg_stmt = msg_stmt.where(AgentMessage.run_id == run_id)
    actions, total = paginate(db, stmt.order_by(AgentAction.created_at.desc()), limit, offset)
    messages = db.scalars(msg_stmt.order_by(AgentMessage.created_at.desc()).limit(limit)).all()
    return {
        "agent_id": agent_id,
        "actions": {"items": [{
            "id": a.id, "run_id": a.run_id, "experiment_id": a.experiment_id, "action_type": a.action_type,
            "status": a.status, "output": a.output, "error": a.error, "retrieved_memory_ids": a.retrieved_memory_ids,
            "retrieval_count": a.retrieval_count, "retrieval_ms": a.retrieval_ms, "duration_ms": a.duration_ms,
            "created_at": a.created_at.isoformat()} for a in actions], "total": total, "limit": limit,
            "offset": offset},
        "messages": [{"id": m.id, "run_id": m.run_id, "sender": m.sender, "recipient": m.recipient,
                      "message_type": m.message_type, "experiment_id": m.experiment_id, "payload": m.payload,
                      "created_at": m.created_at.isoformat()} for m in messages],
    }
