"""Metadata definitions for ModelDoctor components."""

from typing import List

from pydantic import BaseModel, Field


class DoctorMetadata(BaseModel):
    """Metadata describing a Doctor module's capabilities and requirements.

    Attributes:
        name: Unique identifier for the doctor.
        priority: Execution priority (lower runs earlier).
        dimension: The health dimension this doctor evaluates.
        supported_tasks: List of TaskType string values supported by this doctor.
        supported_model_types: List of model families supported (e.g., 'tree', 'linear').
        optional: If True, failure to run won't affect the overall pipeline success.
        estimated_runtime: 'fast', 'medium', or 'slow'.
        dependencies: List of external package names required to run.
    """

    name: str
    priority: int = 50
    dimension: str = "General"
    supported_tasks: List[str] = Field(default_factory=list)
    supported_model_types: List[str] = Field(default_factory=list)
    optional: bool = False
    estimated_runtime: str = "fast"
    dependencies: List[str] = Field(default_factory=list)
