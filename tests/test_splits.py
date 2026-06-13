"""Tests for official WSTS year folds."""

from pathlib import Path

from wildfire_alert.data.event_split import make_event_kfolds
from wildfire_alert.data.splits import WSTS_YEARS, official_year_folds
from wildfire_alert.data.synthetic import create_synthetic_dataset


def test_official_folds_cover_all_validation_test_permutations() -> None:
    folds = official_year_folds()
    assert len(folds) == 12
    assert len({(fold.validation_year, fold.test_year) for fold in folds}) == 12


def test_official_folds_have_no_year_leakage() -> None:
    for fold in official_year_folds():
        train = set(fold.train_years)
        validation = {fold.validation_year}
        test = {fold.test_year}
        assert train.isdisjoint(validation)
        assert train.isdisjoint(test)
        assert validation.isdisjoint(test)
        assert train | validation | test == set(WSTS_YEARS)


def test_event_kfolds_have_no_validation_event_overlap(tmp_path: Path) -> None:
    create_synthetic_dataset(tmp_path, years=(2019,), events_per_year=9)
    folds = make_event_kfolds(tmp_path, years=(2019,), backend="npz", n_splits=3, seed=123)

    validation_sets = [fold.validation_event_ids for fold in folds]
    assert all(len(validation) == 3 for validation in validation_sets)
    for index, validation in enumerate(validation_sets):
        assert validation.isdisjoint(folds[index].train_event_ids)
        for other in validation_sets[index + 1 :]:
            assert validation.isdisjoint(other)
