from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest

from vast_mcp.models import Config, Instance, Service
from vast_mcp.monitor import Monitor
from vast_mcp.state import StateManager


@pytest.fixture
def monitor_deps(tmp_state_dir):
    sm = StateManager(tmp_state_dir)
    sm.save_config(Config(api_key="test-key", idle_threshold_hours=10))
    mock_client = MagicMock()
    return sm, mock_client


class TestSyncStatus:
    def test_updates_status_from_api(self, monitor_deps):
        sm, mock_client = monitor_deps
        inst = Instance(instance_id=1, machine_id=2, gpu_name="A100", num_gpus=1, image="img", status="running")
        sm.save_instance(inst)
        mock_client.show_instances.return_value = [
            {"id": 1, "actual_status": "stopped"}
        ]
        mon = Monitor(sm, mock_client)
        mon.sync_status()
        loaded = sm.load_instances()
        assert loaded[1].status == "stopped"

    def test_removes_gone_instances(self, monitor_deps):
        sm, mock_client = monitor_deps
        inst = Instance(instance_id=1, machine_id=2, gpu_name="A100", num_gpus=1, image="img", status="running")
        sm.save_instance(inst)
        mock_client.show_instances.return_value = []  # Instance gone
        mon = Monitor(sm, mock_client)
        with patch("vast_mcp.monitor.notify") as mock_notify:
            removed = mon.sync_status()
            assert 1 in removed
            mock_notify.assert_called_once()
        loaded = sm.load_instances()
        assert 1 not in loaded


class TestCheckLiveness:
    def test_marks_alive_service(self, monitor_deps, monkeypatch):
        sm, mock_client = monitor_deps
        svc = Service(name="api", port=8000, endpoint="http://x:8000/health", summary="API", last_alive=None)
        inst = Instance(instance_id=1, machine_id=2, gpu_name="A100", num_gpus=1, image="img", status="running", services=[svc])
        sm.save_instance(inst)
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        monkeypatch.setattr("vast_mcp.monitor.httpx.get", lambda *a, **kw: mock_resp)
        mon = Monitor(sm, mock_client)
        mon.check_liveness()
        loaded = sm.load_instances()
        assert loaded[1].services[0].last_alive is not None


class TestEvaluateIdle:
    def test_notifies_when_all_services_idle(self, monitor_deps):
        sm, mock_client = monitor_deps
        old_time = (datetime.now(timezone.utc) - timedelta(hours=11)).isoformat()
        svc = Service(name="api", port=8000, summary="API", last_alive=old_time)
        inst = Instance(instance_id=1, machine_id=2, gpu_name="A100", num_gpus=2, image="img", status="running", services=[svc])
        sm.save_instance(inst)
        mon = Monitor(sm, mock_client)
        with patch("vast_mcp.monitor.notify") as mock_notify:
            mon.evaluate_idle()
            mock_notify.assert_called_once()
            assert "1" in mock_notify.call_args[0][1]

    def test_no_notification_when_recently_alive(self, monitor_deps):
        sm, mock_client = monitor_deps
        recent = datetime.now(timezone.utc).isoformat()
        svc = Service(name="api", port=8000, summary="API", last_alive=recent)
        inst = Instance(instance_id=1, machine_id=2, gpu_name="A100", num_gpus=1, image="img", status="running", services=[svc])
        sm.save_instance(inst)
        mon = Monitor(sm, mock_client)
        with patch("vast_mcp.monitor.notify") as mock_notify:
            mon.evaluate_idle()
            mock_notify.assert_not_called()

    def test_no_services_no_notification(self, monitor_deps):
        sm, mock_client = monitor_deps
        inst = Instance(instance_id=1, machine_id=2, gpu_name="A100", num_gpus=1, image="img", status="running")
        sm.save_instance(inst)
        mon = Monitor(sm, mock_client)
        with patch("vast_mcp.monitor.notify") as mock_notify:
            mon.evaluate_idle()
            mock_notify.assert_not_called()
