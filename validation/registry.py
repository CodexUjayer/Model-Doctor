"""Scenario Registry and Discovery."""

import importlib
import inspect
import pkgutil
from typing import List, Type

import validation.scenarios
from validation.scenario import ValidationScenario

class ScenarioRegistry:
    """Discovers and holds Validation Scenarios."""

    def __init__(self):
        self.scenarios: List[Type[ValidationScenario]] = []

    def discover(self):
        """Automatically discovers all ValidationScenario subclasses in validation/scenarios/."""
        self.scenarios.clear()
        
        # We need to iterate over all subpackages in validation.scenarios
        for _, name, is_pkg in pkgutil.walk_packages(validation.scenarios.__path__, validation.scenarios.__name__ + "."):
            if not is_pkg:
                try:
                    module = importlib.import_module(name)
                    for attr_name in dir(module):
                        attr = getattr(module, attr_name)
                        if (
                            inspect.isclass(attr)
                            and issubclass(attr, ValidationScenario)
                            and attr is not ValidationScenario
                        ):
                            # Ensure we don't add the same class twice if imported in multiple places
                            if attr not in self.scenarios:
                                self.scenarios.append(attr)
                except Exception as e:
                    print(f"Error loading module {name}: {e}")

    def get_all(self) -> List[Type[ValidationScenario]]:
        return self.scenarios
