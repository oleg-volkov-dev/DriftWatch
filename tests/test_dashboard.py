from io import BytesIO, StringIO
from unittest.mock import MagicMock, patch

from infra.dashboard import server


def handler():
    instance = object.__new__(server.Handler)
    instance.send_response = MagicMock()
    instance.send_header = MagicMock()
    instance.end_headers = MagicMock()
    instance.wfile = BytesIO()
    return instance


def test_dashboard_serves_single_html_source():
    instance = handler()
    instance._html()
    assert instance.wfile.getvalue() == server.HTML_PATH.read_bytes()


def test_second_pipeline_is_rejected_while_first_runs():
    instance = handler()
    with server.COMMAND_LOCK:
        instance._stream("train")
    instance.send_response.assert_called_once_with(409)


def test_disconnected_browser_does_not_abandon_process():
    instance = handler()
    instance._sse = MagicMock(side_effect=BrokenPipeError)
    proc = MagicMock(stdout=StringIO("first\nsecond\n"), returncode=0)
    with patch.object(server.subprocess, "Popen", return_value=proc):
        instance._stream("train")
    proc.wait.assert_called()
    assert proc.stdout.closed
    assert not server.COMMAND_LOCK.locked()


def test_failed_process_start_reports_failure_and_releases_lock():
    instance = handler()
    with patch.object(server.subprocess, "Popen", side_effect=FileNotFoundError("make")):
        instance._stream("train")
    assert b'"exit": 1' in instance.wfile.getvalue()
    assert not server.COMMAND_LOCK.locked()
