"""BaseRenderer — abstract base class for report renderers."""

import abc
from pathlib import Path

from modeldoctor.models.report import Report


class BaseRenderer(abc.ABC):
    """Abstract base class for report renderers."""

    @abc.abstractmethod
    def render(self, report: Report, output_path: str | Path | None = None) -> str:
        """Render a Report object to a string and optionally save to disk.
        
        Args:
            report: The generated report object.
            output_path: Optional path to save the rendered report.
            
        Returns:
            The rendered report content as a string.
        """
        ...
