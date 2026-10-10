import pytest

from config.db.schema import Prediction


@pytest.mark.parametrize(("level", "value"), [("empty", 0.0), ("half", 0.5), ("full", 1.0)])
def test_stock_value_follows_stock_level(level, value):
    assert Prediction(stock_level=level).stock_value == value
