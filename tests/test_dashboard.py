import pytest

from upi import config

streamlit_testing = pytest.importorskip("streamlit.testing.v1")


@pytest.mark.skipif(not config.DB_PATH.exists(), reason="run `make analytics` first")
def test_dashboard_renders_without_errors():
    app = streamlit_testing.AppTest.from_file(str(config.ROOT / "app" / "dashboard.py"), default_timeout=120)
    app.run()
    assert not app.exception, app.exception
    assert len(app.metric) >= 4
    # switch every selector to each option once
    for box in app.selectbox:
        for option in box.options:
            box.select(option).run()
            assert not app.exception, (box.label, option, app.exception)
