"""Testing the second order low pass filter."""

from math import isclose

from pandas import read_csv

from hip_controller.definitions import FilterConfig
from hip_controller.utils.low_pass_filter import SecondOrderLPF
from tests.conftest import (
    DATA_SECOND_ORDER_LOW_PASS_FILTER,
    KinematicsDataColumnName,
)


def test_second_order_lpf() -> None:
    """Test the calculation of second order low pass filter.

    Comparing my filter output value of current timestamp, to the Simulink output value of the next timestamp (after 0.01s), they are quite the same with an absolute tolerance of 0.001 for y and 0.005 for yd.
    :return: None
    """
    df = read_csv(filepath_or_buffer=DATA_SECOND_ORDER_LOW_PASS_FILTER)
    config = FilterConfig(wn=20.0, zt=1, x0=0)
    low_pass_filter = SecondOrderLPF(config=config)

    n = len(df)
    for i in range(1, n - 1):
        prev = df.iloc[i - 1]
        curr = df.iloc[i]
        next = df.iloc[i + 1]

        curr_timestamp = curr[KinematicsDataColumnName.TIMESTAMP]
        prev_timestamp = prev[KinematicsDataColumnName.TIMESTAMP]
        input_x = curr["x"]
        expected_y = next["y"]
        expected_yd = next["yd"]

        # Act
        output_y, output_yd = low_pass_filter.step(
            x=input_x, dt=curr_timestamp - prev_timestamp
        )

        # Assert

        assert isclose(output_y, expected_y, abs_tol=0.001), f"Row {i}"
        assert isclose(output_yd, expected_yd, abs_tol=0.005), f"Row {i}"
