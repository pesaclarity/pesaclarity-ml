"""The default Kenyan category set, loaded from ``resources/categories.yaml``."""

from __future__ import annotations

from enum import StrEnum
from functools import cache
from importlib import resources
from typing import Self

import yaml
from pydantic import BaseModel, ConfigDict, model_validator


class CategoryType(StrEnum):
    INCOME = "income"
    EXPENSE = "expense"
    DEBT = "debt"
    SAVINGS = "savings"
    TAX = "tax"
    TRANSFER = "transfer"


class Category(BaseModel):
    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    type: CategoryType
    daily: bool
    target: bool


class CategorySet(BaseModel):
    model_config = ConfigDict(frozen=True)

    version: int
    fallback: str
    categories: tuple[Category, ...]

    @model_validator(mode="after")
    def _check(self) -> Self:
        ids = [c.id for c in self.categories]
        if len(ids) != len(set(ids)):
            raise ValueError("category ids must be unique")
        if self.fallback not in ids:
            raise ValueError(f"fallback {self.fallback!r} is not a category")
        return self

    @property
    def labels(self) -> tuple[str, ...]:
        """Ids the categoriser predicts, in output-layer order."""
        return tuple(c.id for c in self.categories if c.target)

    def index(self, category_id: str) -> int:
        """Output-layer index of a target category."""
        try:
            return self.labels.index(category_id)
        except ValueError as exc:
            raise KeyError(f"{category_id!r} is not a categoriser target") from exc

    def get(self, category_id: str) -> Category:
        for c in self.categories:
            if c.id == category_id:
                return c
        raise KeyError(category_id)


@cache
def default_categories() -> CategorySet:
    """Load the packaged default category set."""
    text = resources.files("akili.resources").joinpath("categories.yaml").read_text("utf-8")
    return CategorySet.model_validate(yaml.safe_load(text))
