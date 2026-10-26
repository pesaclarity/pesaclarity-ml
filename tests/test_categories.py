import pytest

from akili.categories import CategorySet, CategoryType, default_categories
from akili.synthetic import PROFILES


def test_default_set_loads_with_targets_in_file_order() -> None:
    cats = default_categories()
    assert cats.labels[0] == "salary"
    assert "own_transfer" not in cats.labels
    assert cats.fallback == "uncategorised"
    assert cats.fallback not in cats.labels
    assert cats.index("groceries") == cats.labels.index("groceries")


def test_every_target_has_a_synthetic_profile() -> None:
    assert set(default_categories().labels) == set(PROFILES)


def test_category_types_cover_the_app_data_model() -> None:
    types = {c.type for c in default_categories().categories}
    assert types == set(CategoryType)


def test_unknown_target_raises() -> None:
    with pytest.raises(KeyError):
        default_categories().index("own_transfer")


def test_duplicate_ids_rejected() -> None:
    cat = {"id": "a", "name": "A", "type": "expense", "daily": False, "target": True}
    with pytest.raises(ValueError, match="unique"):
        CategorySet.model_validate({"version": 1, "fallback": "a", "categories": [cat, cat]})
