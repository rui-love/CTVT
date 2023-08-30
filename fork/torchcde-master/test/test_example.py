import pathlib
import sys

_here = pathlib.Path(__file__).resolve().parent
sys.path.append(str(_here / "../example"))

import irregular_data  # noqa: E402
import logsignature_example  # noqa: E402
import time_series_classification  # noqa: E402

import markers  # noqa: E402


def test_irregular_data():
    irregular_data.irregular_data()


def test_time_series_classification():
    time_series_classification.main(num_epochs=3)


@markers.uses_signatory
def test_logsignature_example():
    logsignature_example.main(num_epochs=1)


if __name__ == "__main__":
    print(__file__)
    """
    print("Testing irregular data example...")
    test_irregular_data()
    print("Testing time series classification example...")
    test_time_series_classification()
    print("Testing logsignature example...")
    test_logsignature_example()
    """
