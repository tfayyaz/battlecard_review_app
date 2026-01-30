"""
Linked ID System for Battlecard Generation, Evaluation, and Review

This module defines the ID relationships that connect all components of the
battlecard pipeline:

    MLflow Prompt Registry    Unity Catalog Tables     MLflow Experiments
    ─────────────────────    ─────────────────────    ──────────────────
    prompt_name              battlecard_id            experiment_id
    prompt_version           battlecard_version_id    agent_run_id
                             prompt_name
                             prompt_version
                             experiment_id
                             agent_run_id
                             human_review_id

How IDs connect:
- A **prompt** (prompt_name + prompt_version) is registered in MLflow Prompt Registry
- A **generation run** (agent_run_id) uses a specific prompt version to produce output
- The output is stored as a **battlecard** (battlecard_id) in Unity Catalog
- Each generation increments the **battlecard_version_id** for that competitor+product_area
- The MLflow experiment (experiment_id) groups all runs for a given eval context
- **Human reviews** (human_review_id) link back to battlecard_id + agent_run_id + prompt info
"""

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


def new_id() -> str:
    """Generate a new UUID string."""
    return str(uuid.uuid4())


@dataclass
class GenerationIDs:
    """
    All IDs produced during a single battlecard generation run.

    These IDs are written to Unity Catalog and logged to MLflow so that
    every artifact (prompt, generation output, review) can be traced back
    to each other.
    """

    # Unity Catalog identifiers
    battlecard_id: str = field(default_factory=new_id)
    battlecard_version_id: int = 1  # auto-incremented per competitor+product_area

    # MLflow Prompt Registry identifiers
    prompt_name: str = ""  # e.g. "main.default.l200_slide_v1"
    prompt_version: int = 0  # version number from registry
    prompt_alias: str = "latest"  # alias used to load the prompt

    # MLflow Experiment identifiers
    experiment_id: str = ""  # MLflow experiment ID
    agent_run_id: str = ""  # MLflow run ID (set after run starts)

    # Generation metadata
    model_name: str = ""
    competitor: str = ""
    product_area: str = ""
    session_id: str = ""  # Links two-pass generation runs to the same session
    generated_at: datetime = field(default_factory=datetime.utcnow)

    def to_dict(self) -> dict:
        """Serialize all IDs for storage."""
        return {
            "battlecard_id": self.battlecard_id,
            "battlecard_version_id": self.battlecard_version_id,
            "prompt_name": self.prompt_name,
            "prompt_version": self.prompt_version,
            "prompt_alias": self.prompt_alias,
            "experiment_id": self.experiment_id,
            "agent_run_id": self.agent_run_id,
            "model_name": self.model_name,
            "competitor": self.competitor,
            "product_area": self.product_area,
            "session_id": self.session_id,
            "generated_at": self.generated_at.isoformat(),
        }

    def to_mlflow_params(self) -> dict:
        """Return params suitable for mlflow.log_params()."""
        return {
            "battlecard_id": self.battlecard_id,
            "battlecard_version_id": self.battlecard_version_id,
            "prompt_name": self.prompt_name,
            "prompt_version": self.prompt_version,
            "competitor": self.competitor,
            "product_area": self.product_area,
            "model_name": self.model_name,
            "session_id": self.session_id,
        }


@dataclass
class ReviewIDs:
    """
    IDs for a human review session, linking back to generation artifacts.
    """

    human_review_id: str = field(default_factory=new_id)

    # Links back to generation
    battlecard_id: str = ""
    battlecard_version_id: int = 0
    agent_run_id: str = ""
    prompt_name: str = ""
    prompt_version: int = 0
    experiment_id: str = ""

    # Review metadata
    reviewer_email: str = ""
    reviewed_at: datetime = field(default_factory=datetime.utcnow)

    @classmethod
    def from_generation(
        cls,
        gen_ids: GenerationIDs,
        reviewer_email: str,
    ) -> "ReviewIDs":
        """Create ReviewIDs linked to a generation run."""
        return cls(
            battlecard_id=gen_ids.battlecard_id,
            battlecard_version_id=gen_ids.battlecard_version_id,
            agent_run_id=gen_ids.agent_run_id,
            prompt_name=gen_ids.prompt_name,
            prompt_version=gen_ids.prompt_version,
            experiment_id=gen_ids.experiment_id,
            reviewer_email=reviewer_email,
        )

    def to_dict(self) -> dict:
        """Serialize all IDs for storage."""
        return {
            "human_review_id": self.human_review_id,
            "battlecard_id": self.battlecard_id,
            "battlecard_version_id": self.battlecard_version_id,
            "agent_run_id": self.agent_run_id,
            "prompt_name": self.prompt_name,
            "prompt_version": self.prompt_version,
            "experiment_id": self.experiment_id,
            "reviewer_email": self.reviewer_email,
            "reviewed_at": self.reviewed_at.isoformat(),
        }
