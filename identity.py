"""Identity and attribution helpers for user/agent metadata stamping."""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

from sqlalchemy import text

DEFAULT_USER_ID = "anonymous_user"
DEFAULT_USER_NAME = "Anonymous User"
DEFAULT_USER_EMAIL = "anonymous@databricks.local"
DEFAULT_USER_TEAM = "unknown"

DEFAULT_AGENT_ID = "battlecard_review_app"
DEFAULT_AGENT_NAME = "Battlecard Review App"

ROLE_ARTIFACT_CREATOR = "ARTIFACT_CREATOR"
ROLE_ARTIFACT_REVIEWER = "ARTIFACT_REVIEWER"


def _sanitize_identifier(raw: Any, fallback: str) -> str:
    value = str(raw or "").strip()
    if not value:
        return fallback
    value = re.sub(r"[^A-Za-z0-9_.:@-]+", "_", value)
    return value[:255] or fallback


def extract_request_user(request_obj) -> dict[str, str]:
    """Extract best-effort user identity from request headers/query args."""
    headers = getattr(request_obj, "headers", {}) or {}
    args = getattr(request_obj, "args", {}) or {}

    email = (
        headers.get("X-User-Email")
        or headers.get("X-Forwarded-Email")
        or headers.get("X-Forwarded-User")
        or headers.get("X-Databricks-User")
        or args.get("user_email")
        or DEFAULT_USER_EMAIL
    )
    email = str(email).strip() or DEFAULT_USER_EMAIL

    user_id_raw = headers.get("X-User-Id") or args.get("user_id")
    user_id = _sanitize_identifier(user_id_raw or email.lower(), DEFAULT_USER_ID)

    user_name = (
        headers.get("X-User-Name")
        or headers.get("X-Forwarded-Name")
        or args.get("user_name")
        or email.split("@")[0]
        or DEFAULT_USER_NAME
    )
    user_team = headers.get("X-User-Team") or args.get("user_team") or DEFAULT_USER_TEAM

    return {
        "user_id": user_id,
        "user_name": str(user_name).strip()[:255] or DEFAULT_USER_NAME,
        "user_email": email[:255],
        "user_team": str(user_team).strip()[:255] or DEFAULT_USER_TEAM,
    }


def system_user_context(
    user_id: str = DEFAULT_USER_ID,
    user_name: str = DEFAULT_USER_NAME,
    user_email: str = DEFAULT_USER_EMAIL,
    user_team: str = DEFAULT_USER_TEAM,
) -> dict[str, str]:
    return {
        "user_id": _sanitize_identifier(user_id, DEFAULT_USER_ID),
        "user_name": str(user_name or DEFAULT_USER_NAME)[:255],
        "user_email": str(user_email or DEFAULT_USER_EMAIL)[:255],
        "user_team": str(user_team or DEFAULT_USER_TEAM)[:255],
    }


def ensure_user(conn, user_ctx: Optional[Mapping[str, Any]]) -> str:
    ctx = dict(user_ctx or {})
    user_id = _sanitize_identifier(ctx.get("user_id"), DEFAULT_USER_ID)
    user_name = str(ctx.get("user_name") or DEFAULT_USER_NAME)[:255]
    user_email = str(ctx.get("user_email") or DEFAULT_USER_EMAIL)[:255]
    user_team = str(ctx.get("user_team") or DEFAULT_USER_TEAM)[:255]

    existing_by_email = None
    if user_email:
        existing_by_email = conn.execute(
            text("SELECT user_id FROM users WHERE user_email = :email LIMIT 1"),
            {"email": user_email},
        ).scalar()
    if existing_by_email and str(existing_by_email) != user_id:
        user_id = str(existing_by_email)

    conn.execute(
        text(
            "INSERT INTO users (user_id, user_name, user_email, user_team, is_active) "
            "VALUES (:uid, :name, :email, :team, TRUE) "
            "ON CONFLICT (user_id) DO UPDATE SET "
            "user_name = EXCLUDED.user_name, "
            "user_email = EXCLUDED.user_email, "
            "user_team = EXCLUDED.user_team, "
            "updated_at = NOW(), "
            "is_active = TRUE"
        ),
        {"uid": user_id, "name": user_name, "email": user_email, "team": user_team},
    )
    return user_id


def ensure_role(conn, role_name: str, feature_flags: Optional[dict[str, Any]] = None) -> int:
    row = conn.execute(
        text(
            "INSERT INTO roles (role_name, feature_flags, is_active) "
            "VALUES (:name, CAST(:flags AS jsonb), TRUE) "
            "ON CONFLICT (role_name) DO UPDATE SET "
            "feature_flags = EXCLUDED.feature_flags, "
            "is_active = TRUE, "
            "updated_at = NOW() "
            "RETURNING role_id"
        ),
        {"name": role_name, "flags": "{}" if feature_flags is None else json_dumps(feature_flags)},
    ).mappings().first()
    return int(row["role_id"])


def ensure_user_role(conn, user_id: str, role_name: str, assigned_by_user_id: Optional[str] = None) -> int:
    role_id = ensure_role(conn, role_name)
    conn.execute(
        text(
            "INSERT INTO user_assigned_roles (user_id, role_id, assigned_by_user_id) "
            "VALUES (:uid, :rid, :assigned_by) "
            "ON CONFLICT (user_id, role_id) DO NOTHING"
        ),
        {"uid": user_id, "rid": role_id, "assigned_by": assigned_by_user_id},
    )
    return role_id


def ensure_agent(
    conn,
    agent_id: str,
    agent_name: str,
    *,
    agent_version: str = "",
    llm_model: str = "",
    llm_model_version: str = "",
    agent_tools_list: Optional[list[str]] = None,
) -> str:
    clean_agent_id = _sanitize_identifier(agent_id, DEFAULT_AGENT_ID)
    clean_agent_name = str(agent_name or DEFAULT_AGENT_NAME)[:255]
    tools_json = json_dumps(agent_tools_list or [])
    conn.execute(
        text(
            "INSERT INTO agents (agent_id, agent_name, agent_version, llm_model, llm_model_version, "
            "agent_tools_list, is_active) "
            "VALUES (:aid, :name, :aver, :model, :mver, CAST(:tools AS jsonb), TRUE) "
            "ON CONFLICT (agent_id) DO UPDATE SET "
            "agent_name = EXCLUDED.agent_name, "
            "agent_version = EXCLUDED.agent_version, "
            "llm_model = EXCLUDED.llm_model, "
            "llm_model_version = EXCLUDED.llm_model_version, "
            "agent_tools_list = EXCLUDED.agent_tools_list, "
            "is_active = TRUE, "
            "updated_at = NOW()"
        ),
        {
            "aid": clean_agent_id,
            "name": clean_agent_name,
            "aver": str(agent_version or "")[:100],
            "model": str(llm_model or "")[:255],
            "mver": str(llm_model_version or "")[:100],
            "tools": tools_json,
        },
    )
    return clean_agent_id


def json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, separators=(",", ":"), ensure_ascii=True)
