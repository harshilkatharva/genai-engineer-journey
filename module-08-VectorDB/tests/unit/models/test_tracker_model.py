from datetime import datetime

from semantic_search_eng.models.tracker_model import TrackerModel


def test_tracker_model_creates_timestamp() -> None:
    tracker = TrackerModel()

    assert isinstance(
        tracker.timestamp,
        datetime,
    )
