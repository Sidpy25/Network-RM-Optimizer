import os

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = os.path.join(os.path.dirname(__file__), "..", "app.py")


def test_dashboard_runs_on_sample_data():
    at = AppTest.from_file(APP, default_timeout=300).run()
    assert not at.exception
    assert not at.error
    assert [m.label for m in at.metric][:3] == ["Revenue", "Contribution", "Net profit"]


def test_upload_mode_waits_for_files():
    at = AppTest.from_file(APP, default_timeout=300).run()
    at.sidebar.radio[0].set_value("Upload my data").run()
    assert not at.exception
    assert any("Upload last year's" in i.value for i in at.info)
