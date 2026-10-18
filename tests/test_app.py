from pathlib import Path

import pytest

pytest.importorskip("streamlit")
from streamlit.testing.v1 import AppTest  # noqa: E402

APP = str(Path(__file__).resolve().parent.parent / "app.py")


@pytest.mark.parametrize("preset", [None, "Lower respiratory tract infection", "Custom symptom timeline"])
def test_app_runs_without_errors(preset):
    app = AppTest.from_file(APP, default_timeout=300)
    app.run()
    if preset:
        app.selectbox[0].select(preset).run()
    assert not app.exception
    assert len(app.metric) == 5
