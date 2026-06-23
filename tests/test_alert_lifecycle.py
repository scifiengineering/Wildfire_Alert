import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd


def load_alert_lifecycle_module():
    script_path = Path(__file__).parents[1] / "scripts" / "38_summarize_alert_lifecycle.py"
    spec = importlib.util.spec_from_file_location("alert_lifecycle", script_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_lifecycle_keeps_event_days_with_missing_distance(tmp_path: Path) -> None:
    module = load_alert_lifecycle_module()
    path = tmp_path / "scored_candidates.csv"
    pd.DataFrame(
        [
            {
                "event_id": "fire_with_missing_distance",
                "target_date": "2020-11-25",
                "row": 1,
                "col": 2,
                "model_score": 0.9,
                "current_fire_at_candidate": 0,
                "distance_to_current_fire_km": np.nan,
                "label": 0,
            }
        ]
    ).to_csv(path, index=False)

    candidates = module.load_candidates(path)
    rows = module.summarize_alert_lifecycle(candidates, threshold=0.5, max_radius_km=4.0)
    summary = module.summarize(rows)

    assert summary["event_count"] == 1
    assert summary["event_day_count"] == 1
    assert rows[0]["eligible_candidate_count"] == 0
    assert rows[0]["selected_alert_count"] == 0


def test_rolling_lifecycle_uses_window_step_rows() -> None:
    module = load_alert_lifecycle_module()
    candidates = pd.DataFrame(
        [
            {
                "event_id": "fire_1",
                "target_date": target_date,
                "row": 1,
                "col": 2,
                "model_score": 0.9,
                "current_fire_at_candidate": 0,
                "distance_to_current_fire_km": 1.0,
            }
            for target_date in ["2020-01-01", "2020-01-02", "2020-01-03"]
        ]
    )

    rows = module.summarize_rolling_alert_lifecycle(
        candidates,
        threshold=0.5,
        radii_km=(2.0, 3.0, 4.0),
    )
    summary = module.summarize(rows)

    assert summary["event_day_count"] == 3
    assert summary["window_count"] == 1
    assert summary["mean_active_alerts_per_event_day"] == 1.0
