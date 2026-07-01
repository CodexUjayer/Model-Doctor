"""Knowledge Base for prescriptions and diagnoses."""

import os
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

from modeldoctor.models.enums import Confidence
from modeldoctor.utils.logging import get_logger

logger = get_logger(__name__)


class KnowledgeEntry(BaseModel):
    """A reusable knowledge entry for a specific ML issue."""

    id: str
    applicable_model_families: List[str] = Field(default_factory=list)
    applicable_tasks: List[str] = Field(default_factory=list)
    issue_category: str
    diagnosis: str
    prescription: str
    rationale: str
    references: List[str] = Field(default_factory=list)
    confidence_level: Confidence = Confidence.MEDIUM


class KnowledgeBase:
    """Loads and serves ML diagnostic knowledge entries."""

    def __init__(self, kb_dir: str | Path | None = None) -> None:
        self.entries: Dict[str, KnowledgeEntry] = {}
        if kb_dir is None:
            kb_dir = Path(__file__).parent / "knowledge"
        self._load_knowledge(Path(kb_dir))

    def _load_knowledge(self, kb_dir: Path) -> None:
        if not kb_dir.exists():
            # Create default dir if not exists but don't error
            try:
                kb_dir.mkdir(parents=True, exist_ok=True)
            except Exception:
                pass
            return

        for file in kb_dir.glob("*.yaml"):
            try:
                with open(file, "r") as f:
                    data = yaml.safe_load(f)
                    if isinstance(data, list):
                        for entry_data in data:
                            entry = KnowledgeEntry.parse_obj(entry_data)
                            self.entries[entry.id] = entry
            except Exception as e:
                logger.error(f"Failed to load knowledge from {file}: {e}")

        logger.info(f"Loaded {len(self.entries)} knowledge entries.")

    def get_entry(self, entry_id: str) -> Optional[KnowledgeEntry]:
        """Get a knowledge entry by ID."""
        return self.entries.get(entry_id)
