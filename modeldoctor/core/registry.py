"""Doctor registry — plugin discovery and management.

Doctors can be registered via:

1. **Entry points**: Third-party packages list their doctors under the
   ``modeldoctor.doctors`` entry-point group in their ``pyproject.toml``.
2. **Manual registration**: ``registry.register(MyDoctor)`` at runtime.
3. **Built-in discovery**: ``DoctorRegistry.default()`` loads all built-in
   doctors automatically.

Example entry-point declaration in a third-party package::

    [project.entry-points."modeldoctor.doctors"]
    my_doctor = "mypkg.doctors:MyDoctor"
"""

from __future__ import annotations

import importlib
from typing import Dict, List, Optional, Type

from modeldoctor.core.base_doctor import BaseDoctor
from modeldoctor.utils.logging import get_logger

logger = get_logger(__name__)

_ENTRY_POINT_GROUP = "modeldoctor.doctors"


class DoctorRegistry:
    """Registry for Doctor classes.

    Maintains an ordered mapping of doctor name → class, and supports
    loading from installed package entry points.
    """

    def __init__(self) -> None:
        self._doctors: Dict[str, Type[BaseDoctor]] = {}

    def register(self, doctor_cls: Type[BaseDoctor]) -> None:
        """Register a Doctor class.

        Args:
            doctor_cls: A concrete subclass of :class:`BaseDoctor`.

        Raises:
            TypeError: If *doctor_cls* is not a BaseDoctor subclass.
            ValueError: If a doctor with the same name is already registered.
        """
        if not (isinstance(doctor_cls, type) and issubclass(doctor_cls, BaseDoctor)):
            raise TypeError(f"{doctor_cls} is not a BaseDoctor subclass.")
        name = doctor_cls.name
        if name in self._doctors:
            logger.debug("Overwriting existing doctor registration: '%s'.", name)
        self._doctors[name] = doctor_cls
        logger.debug("Registered doctor: '%s' (%s).", name, doctor_cls.__qualname__)

    def unregister(self, name: str) -> None:
        """Remove a doctor from the registry by name."""
        self._doctors.pop(name, None)

    def get(self, name: str) -> Optional[Type[BaseDoctor]]:
        """Return the doctor class for *name*, or ``None``."""
        return self._doctors.get(name)

    def list_names(self) -> List[str]:
        """Return all registered doctor names."""
        return list(self._doctors.keys())

    def instantiate_all(
        self,
        include: Optional[List[str]] = None,
        exclude: Optional[List[str]] = None,
    ) -> List[BaseDoctor]:
        """Instantiate and return all registered doctors, sorted by priority.

        Args:
            include: Whitelist of doctor names to include (empty = all).
            exclude: Doctor names to skip.

        Returns:
            Sorted list of :class:`BaseDoctor` instances.
        """
        names = list(self._doctors.keys())
        if include:
            names = [n for n in names if n in include]
        if exclude:
            names = [n for n in names if n not in exclude]

        instances = [self._doctors[n]() for n in names]
        return sorted(instances, key=lambda d: d.priority)

    def load_entry_points(self) -> None:
        """Discover and register doctors from installed package entry points.

        Reads the ``modeldoctor.doctors`` entry-point group from all installed
        packages and registers any :class:`BaseDoctor` subclasses found.
        """
        try:
            from importlib.metadata import entry_points

            eps = entry_points(group=_ENTRY_POINT_GROUP)
            for ep in eps:
                try:
                    cls = ep.load()
                    if isinstance(cls, type) and issubclass(cls, BaseDoctor) and cls is not BaseDoctor:
                        self.register(cls)
                        logger.debug("Loaded doctor from entry point: '%s'.", ep.name)
                    else:
                        logger.warning(
                            "Entry point '%s' does not point to a BaseDoctor subclass.",
                            ep.name,
                        )
                except Exception as exc:
                    logger.warning("Failed to load entry point '%s': %s", ep.name, exc)
        except Exception as exc:
            logger.warning("Entry point discovery failed: %s", exc)

    @classmethod
    def default(cls) -> "DoctorRegistry":
        """Create a registry pre-loaded with all built-in doctors.

        This is the factory used by the default diagnostic pipeline.

        Returns:
            A :class:`DoctorRegistry` with all 8 built-in doctors registered.
        """
        registry = cls()
        _load_builtin_doctors(registry)
        registry.load_entry_points()
        return registry

    def __len__(self) -> int:
        return len(self._doctors)

    def __repr__(self) -> str:
        return f"DoctorRegistry(doctors={self.list_names()})"


def _load_builtin_doctors(registry: DoctorRegistry) -> None:
    """Import and register all built-in Doctor classes."""
    builtins = [
        ("modeldoctor.doctors.data_doctor", "DataDoctor"),
        ("modeldoctor.doctors.feature_doctor", "FeatureDoctor"),
        ("modeldoctor.doctors.overfitting_doctor", "OverfittingDoctor"),
        ("modeldoctor.doctors.leakage_doctor", "LeakageDoctor"),
        ("modeldoctor.doctors.hyperparameter_doctor", "HyperparameterDoctor"),
        ("modeldoctor.doctors.prediction_doctor", "PredictionDoctor"),
        ("modeldoctor.doctors.generalization_doctor", "GeneralizationDoctor"),
        ("modeldoctor.doctors.production_doctor", "ProductionDoctor"),
        ("modeldoctor.doctors.calibration", "CalibrationDoctor"),
    ]
    for module_path, class_name in builtins:
        try:
            mod = importlib.import_module(module_path)
            cls = getattr(mod, class_name)
            registry.register(cls)
        except Exception as exc:
            logger.warning("Failed to load built-in doctor %s.%s: %s", module_path, class_name, exc)
