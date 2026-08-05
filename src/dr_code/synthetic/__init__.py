"""Synthetic test-corpus generation.

Public surface:
    - `RECIPES`: frozen tuple of `Recipe` definitions
    - `Recipe`, `apply_recipe`: recipe model + applier
    - `build_dataset`, `build_sample`: dataset builder
    - `save_dataset`, `load_dataset`: JSONL I/O
    - `load_humaneval_plus`, `HumanEvalPlusTask`: ground-truth loader
    - `REGISTRY`, `Corruption`: corruptions
"""

from dr_code.synthetic.corruption_recipes import (
    RECIPES,
    RECIPES_BY_NAME,
    Recipe,
    apply_recipe,
)
from dr_code.synthetic.dataset_builder import (
    build_dataset,
    build_sample,
    load_dataset,
    save_dataset,
)
from dr_code.synthetic.humaneval_loader import (
    HumanEvalPlusTask,
    load_humaneval_plus,
)
from dr_code.synthetic.corruptions import REGISTRY, Corruption

__all__ = [
    "RECIPES",
    "RECIPES_BY_NAME",
    "REGISTRY",
    "HumanEvalPlusTask",
    "Corruption",
    "Recipe",
    "apply_recipe",
    "build_dataset",
    "build_sample",
    "load_dataset",
    "load_humaneval_plus",
    "save_dataset",
]
