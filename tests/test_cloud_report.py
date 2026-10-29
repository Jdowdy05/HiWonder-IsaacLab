import importlib.util
from pathlib import Path

import pytest


def load():
    path = Path(__file__).parents[1] / "cloud/report.py"
    spec = importlib.util.spec_from_file_location("cloud_report", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_cost_uses_end_to_end_fps_and_hourly_price():
    result = load().summarize_samples([10, 100, 100, 100], [2, 0.7, 0.7, 0.7], [1, 0.3, 0.3, 0.3], 0.5, discard=1)
    assert result["median_transitions_per_second"] == 100
    assert result["usd_per_million_transitions"] == pytest.approx(0.5 * 1_000_000 / 360_000)
    assert result["median_learning_fraction"] == pytest.approx(0.3)
    assert result["ideal_total_speedup_if_learner_doubles"] == pytest.approx(1 / 0.85)


@pytest.mark.parametrize("samples,price", (([], 0.5), ([float("nan")], 0.5), ([0], 0.5), ([100], 0)))
def test_missing_or_invalid_measurement_cannot_make_a_value_claim(samples, price):
    with pytest.raises(ValueError):
        load().summarize_samples(samples, [], [], price, discard=0)
