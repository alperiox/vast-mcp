from unittest.mock import patch, MagicMock

from vast_mcp.notifications import notify


class TestNotify:
    def test_calls_osascript(self):
        with patch("vast_mcp.notifications.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            notify("Test Title", "Test message body")
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert cmd[0] == "osascript"
            assert cmd[1] == "-e"
            assert "Test Title" in cmd[2]
            assert "Test message body" in cmd[2]

    def test_handles_osascript_failure(self):
        with patch("vast_mcp.notifications.subprocess.run", side_effect=FileNotFoundError):
            # Should not raise
            notify("Title", "Body")
