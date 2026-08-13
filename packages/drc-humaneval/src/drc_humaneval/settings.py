from __future__ import annotations

from typing import Self

from pydantic import model_validator

from dr_code.metrics.settings import OperatorSettings


class CodeTestSettings(OperatorSettings):
    task_key: str = "task"

    @model_validator(mode="after")
    def validate_values(self) -> Self:
        if not self.task_key:
            raise ValueError("task_key must not be empty")
        return self


__all__ = ["CodeTestSettings"]
