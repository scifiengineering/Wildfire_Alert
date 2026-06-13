"""Official year-based cross-validation assignments for WSTS."""

import json
from dataclasses import dataclass
from itertools import permutations
from pathlib import Path

WSTS_YEARS: tuple[int, ...] = (2018, 2019, 2020, 2021)


@dataclass(frozen=True)
class Fold:
    """One official WSTS train/validation/test year assignment."""

    index: int
    train_years: tuple[int, int]
    validation_year: int
    test_year: int

    def as_dict(self) -> dict[str, object]:
        """Return a JSON-serializable representation."""

        return {
            "index": self.index,
            "train_years": list(self.train_years),
            "validation_year": self.validation_year,
            "test_year": self.test_year,
        }


def official_year_folds(years: tuple[int, ...] = WSTS_YEARS) -> tuple[Fold, ...]:
    """Build all 12 official assignments of validation and test years.

    The two years not selected for validation or testing form the training set.
    """

    if len(years) != 4 or len(set(years)) != 4:
        raise ValueError("WSTS cross-validation requires exactly four unique years.")

    folds: list[Fold] = []
    for index, (validation_year, test_year) in enumerate(permutations(years, 2)):
        train_years = tuple(sorted(set(years) - {validation_year, test_year}))
        folds.append(
            Fold(
                index=index,
                train_years=(train_years[0], train_years[1]),
                validation_year=validation_year,
                test_year=test_year,
            )
        )
    return tuple(folds)


def write_official_folds(path: Path) -> None:
    """Write official fold assignments to JSON."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps([fold.as_dict() for fold in official_year_folds()], indent=2) + "\n",
        encoding="utf-8",
    )
