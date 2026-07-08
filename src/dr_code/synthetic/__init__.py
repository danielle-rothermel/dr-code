"""Synthetic test-corpus generation.

Public surface:
    - `RECIPES`: frozen tuple of `Recipe` definitions
    - `Recipe`, `apply_recipe`: recipe model + applier
    - `build_dataset`, `build_sample`, `iter_dataset`: dataset builder
    - `save_dataset`, `load_dataset`: JSONL I/O
    - `canonicalize`, `equivalent`: semantic equivalence helpers
    - `load_humaneval_plus`, `save_snapshot`, `HumanEvalPlusTask`: ground-truth loader
    - `REGISTRY`, `InverseTransform`: inverse transforms
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
    iter_dataset,
    load_dataset,
    save_dataset,
)
from dr_code.synthetic.equivalence import canonicalize, equivalent
from dr_code.synthetic.humaneval_loader import (
    HumanEvalPlusTask,
    load_humaneval_plus,
    save_snapshot,
)
from dr_code.synthetic.inverse_transforms import REGISTRY, InverseTransform

__all__ = [
    "RECIPES",
    "RECIPES_BY_NAME",
    "REGISTRY",
    "HumanEvalPlusTask",
    "InverseTransform",
    "Recipe",
    "apply_recipe",
    "build_dataset",
    "build_sample",
    "canonicalize",
    "equivalent",
    "iter_dataset",
    "load_dataset",
    "load_humaneval_plus",
    "save_dataset",
    "save_snapshot",
]
