from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from vast_mcp.models import Config, Instance, Service, Experiment, ExperimentSpecs
from vast_mcp.state import StateManager


@pytest.fixture
def server_deps(tmp_state_dir):
    """Set up a state manager and mock vast client for testing MCP tool functions."""
    sm = StateManager(tmp_state_dir)
    sm.save_config(Config(api_key="test-key"))
    mock_client = MagicMock()
    return sm, mock_client


# ---------------------------------------------------------------------------
# Task 5: Discovery & Config
# ---------------------------------------------------------------------------


class TestSearchOffers:
    def test_search_returns_formatted_results(self, server_deps):
        sm, mock_client = server_deps
        mock_client.search_offers.return_value = [
            {
                "id": 1, "gpu_name": "RTX 4090", "num_gpus": 2,
                "gpu_ram": 24000, "cpu_ram": 64000, "disk_space": 200,
                "dph_total": 0.8, "reliability": 0.99,
            },
        ]
        from vast_mcp.mcp_server import _search_offers
        result = _search_offers(sm, mock_client, query="gpu_name=RTX_4090")
        assert "RTX 4090" in result
        assert "0.8" in result

    def test_search_no_results(self, server_deps):
        sm, mock_client = server_deps
        mock_client.search_offers.return_value = []
        from vast_mcp.mcp_server import _search_offers
        result = _search_offers(sm, mock_client, query="gpu_name=NONEXISTENT")
        assert "No offers found" in result


class TestGetOfferDetails:
    def test_returns_offer_info(self, server_deps):
        sm, mock_client = server_deps
        mock_client.get_offer_details.return_value = {
            "id": 1, "gpu_name": "A100", "num_gpus": 4, "gpu_ram": 80000,
            "cpu_cores": 32, "cpu_ram": 128000, "disk_space": 500,
            "dph_total": 3.2, "reliability": 0.98, "geolocation": "US",
            "cuda_max_good": "12.4", "direct_port_count": 5,
            "inet_down": 800, "inet_up": 400,
        }
        from vast_mcp.mcp_server import _get_offer_details
        result = _get_offer_details(mock_client, offer_id=1)
        assert "A100" in result
        assert "4" in result

    def test_returns_not_found(self, server_deps):
        sm, mock_client = server_deps
        mock_client.get_offer_details.return_value = None
        from vast_mcp.mcp_server import _get_offer_details
        result = _get_offer_details(mock_client, offer_id=999)
        assert "not found" in result.lower()


class TestConfigTools:
    def test_get_config(self, server_deps):
        sm, mock_client = server_deps
        from vast_mcp.mcp_server import _get_config
        result = _get_config(sm)
        assert "idle_threshold_hours" in result
        assert "10" in result

    def test_set_config(self, server_deps):
        sm, mock_client = server_deps
        from vast_mcp.mcp_server import _set_config
        result = _set_config(sm, key="idle_threshold_hours", value=8)
        assert "Updated" in result
        config = sm.load_config()
        assert config.idle_threshold_hours == 8

    def test_set_config_invalid_key(self, server_deps):
        sm, mock_client = server_deps
        from vast_mcp.mcp_server import _set_config
        result = _set_config(sm, key="bad_key", value="x")
        assert "Error" in result or "Unknown" in result


# ---------------------------------------------------------------------------
# Task 6: Instance Lifecycle
# ---------------------------------------------------------------------------


class TestCreateInstance:
    def test_create_and_track(self, server_deps):
        sm, mock_client = server_deps
        mock_client.create_instance.return_value = {"new_contract": 555}
        mock_client.show_instances.return_value = [
            {
                "id": 555, "machine_id": 100, "gpu_name": "RTX 4090",
                "num_gpus": 2, "image_uuid": "pytorch/pytorch:latest",
                "actual_status": "running",
            }
        ]
        from vast_mcp.mcp_server import _create_instance
        result = _create_instance(sm, mock_client, offer_id=99, image="pytorch/pytorch:latest", disk_space=50)
        assert "555" in result
        instances = sm.load_instances()
        assert 555 in instances


class TestStopInstance:
    def test_stop_tracked_instance(self, server_deps):
        sm, mock_client = server_deps
        inst = Instance(instance_id=1, machine_id=2, gpu_name="A100", num_gpus=1, image="img", status="running")
        sm.save_instance(inst)
        from vast_mcp.mcp_server import _stop_instance
        result = _stop_instance(sm, mock_client, instance_id=1)
        assert "Stopped" in result or "stopped" in result
        mock_client.stop_instance.assert_called_once_with(1)

    def test_stop_untracked_instance(self, server_deps):
        sm, mock_client = server_deps
        from vast_mcp.mcp_server import _stop_instance
        result = _stop_instance(sm, mock_client, instance_id=999)
        assert "not tracked" in result.lower() or "not found" in result.lower()


class TestStartInstance:
    def test_start_stopped_instance(self, server_deps):
        sm, mock_client = server_deps
        inst = Instance(instance_id=1, machine_id=2, gpu_name="V100", num_gpus=1, image="img", status="stopped")
        sm.save_instance(inst)
        from vast_mcp.mcp_server import _start_instance
        result = _start_instance(sm, mock_client, instance_id=1)
        assert "Started" in result or "started" in result
        mock_client.start_instance.assert_called_once_with(1)


class TestDestroyInstance:
    def test_destroy_removes_from_tracking(self, server_deps):
        sm, mock_client = server_deps
        inst = Instance(instance_id=1, machine_id=2, gpu_name="A100", num_gpus=1, image="img", status="running")
        sm.save_instance(inst)
        from vast_mcp.mcp_server import _destroy_instance
        result = _destroy_instance(sm, mock_client, instance_id=1)
        assert "Destroyed" in result or "destroyed" in result
        instances = sm.load_instances()
        assert 1 not in instances


class TestListInstances:
    def test_list_with_services(self, server_deps):
        sm, mock_client = server_deps
        svc = Service(name="api", port=8000, summary="REST API")
        inst = Instance(
            instance_id=1, machine_id=2, gpu_name="RTX 4090",
            num_gpus=2, image="pytorch", status="running",
            services=[svc],
        )
        sm.save_instance(inst)
        mock_client.show_instances.return_value = [
            {"id": 1, "actual_status": "running"}
        ]
        from vast_mcp.mcp_server import _list_instances
        result = _list_instances(sm, mock_client)
        assert "RTX 4090" in result
        assert "api" in result

    def test_list_empty(self, server_deps):
        sm, mock_client = server_deps
        mock_client.show_instances.return_value = []
        from vast_mcp.mcp_server import _list_instances
        result = _list_instances(sm, mock_client)
        assert "No instances" in result or "no instances" in result


# ---------------------------------------------------------------------------
# Task 7: Service Registry
# ---------------------------------------------------------------------------


class TestRegisterService:
    def test_register_on_tracked_instance(self, server_deps):
        sm, mock_client = server_deps
        inst = Instance(instance_id=1, machine_id=2, gpu_name="A100", num_gpus=1, image="img", status="running")
        sm.save_instance(inst)
        from vast_mcp.mcp_server import _register_service
        result = _register_service(sm, instance_id=1, name="vllm", port=8000, endpoint="http://x:8000/v1", summary="vLLM")
        assert "Registered" in result
        loaded = sm.load_instances()
        assert len(loaded[1].services) == 1
        assert loaded[1].services[0].name == "vllm"

    def test_register_on_untracked_instance(self, server_deps):
        sm, mock_client = server_deps
        from vast_mcp.mcp_server import _register_service
        result = _register_service(sm, instance_id=999, name="api", port=3000, summary="API")
        assert "not tracked" in result.lower()

    def test_register_duplicate_name_replaces(self, server_deps):
        sm, mock_client = server_deps
        inst = Instance(
            instance_id=1, machine_id=2, gpu_name="A100", num_gpus=1, image="img", status="running",
            services=[Service(name="api", port=3000, summary="old")],
        )
        sm.save_instance(inst)
        from vast_mcp.mcp_server import _register_service
        _register_service(sm, instance_id=1, name="api", port=3001, summary="new")
        loaded = sm.load_instances()
        assert len(loaded[1].services) == 1
        assert loaded[1].services[0].port == 3001


class TestUnregisterService:
    def test_unregister_existing(self, server_deps):
        sm, mock_client = server_deps
        inst = Instance(
            instance_id=1, machine_id=2, gpu_name="A100", num_gpus=1, image="img", status="running",
            services=[Service(name="api", port=3000, summary="REST API")],
        )
        sm.save_instance(inst)
        from vast_mcp.mcp_server import _unregister_service
        result = _unregister_service(sm, instance_id=1, service_name="api")
        assert "Removed" in result or "removed" in result
        loaded = sm.load_instances()
        assert len(loaded[1].services) == 0

    def test_unregister_nonexistent(self, server_deps):
        sm, mock_client = server_deps
        inst = Instance(instance_id=1, machine_id=2, gpu_name="A100", num_gpus=1, image="img", status="running")
        sm.save_instance(inst)
        from vast_mcp.mcp_server import _unregister_service
        result = _unregister_service(sm, instance_id=1, service_name="nope")
        assert "not found" in result.lower()


class TestCheckServices:
    def test_check_no_services(self, server_deps):
        sm, mock_client = server_deps
        inst = Instance(instance_id=1, machine_id=2, gpu_name="A100", num_gpus=1, image="img", status="running")
        sm.save_instance(inst)
        from vast_mcp.mcp_server import _check_services
        result = _check_services(sm)
        assert "No services" in result or "no services" in result

    def test_check_reports_service_status(self, server_deps, monkeypatch):
        sm, mock_client = server_deps
        svc = Service(name="api", port=8000, endpoint="http://example.com:8000/health", summary="API")
        inst = Instance(
            instance_id=1, machine_id=2, gpu_name="A100", num_gpus=1,
            image="img", status="running", services=[svc],
        )
        sm.save_instance(inst)
        mock_response = MagicMock()
        mock_response.status_code = 200
        monkeypatch.setattr("vast_mcp.mcp_server.httpx.get", lambda *a, **kw: mock_response)
        from vast_mcp.mcp_server import _check_services
        result = _check_services(sm, instance_id=1)
        assert "up" in result.lower()


# ---------------------------------------------------------------------------
# Task 8: Experiment Templates
# ---------------------------------------------------------------------------


class TestSaveExperiment:
    def test_save_new_experiment(self, server_deps):
        sm, mock_client = server_deps
        from vast_mcp.mcp_server import _save_experiment
        specs = {"gpu_name": "RTX_4090", "num_gpus": 2, "gpu_ram_min": 24000}
        result = _save_experiment(sm, name="my-exp", specs=specs, summary="Test experiment")
        assert "Saved" in result
        loaded = sm.load_experiments()
        assert "my-exp" in loaded
        assert loaded["my-exp"].specs.gpu_name == "RTX_4090"

    def test_save_with_image(self, server_deps):
        sm, mock_client = server_deps
        from vast_mcp.mcp_server import _save_experiment
        _save_experiment(sm, name="img-exp", specs={"gpu_name": "A100"}, summary="With image", image="nvcr.io/nvidia/pytorch:24.01")
        loaded = sm.load_experiments()
        assert loaded["img-exp"].image == "nvcr.io/nvidia/pytorch:24.01"


class TestListExperiments:
    def test_list_empty(self, server_deps):
        sm, mock_client = server_deps
        from vast_mcp.mcp_server import _list_experiments
        result = _list_experiments(sm)
        assert "No experiments" in result or "no experiments" in result

    def test_list_shows_experiments(self, server_deps):
        sm, mock_client = server_deps
        from vast_mcp.mcp_server import _save_experiment, _list_experiments
        _save_experiment(sm, name="exp1", specs={"gpu_name": "A100"}, summary="First")
        _save_experiment(sm, name="exp2", specs={"num_gpus": 4}, summary="Second")
        result = _list_experiments(sm)
        assert "exp1" in result
        assert "exp2" in result


class TestLoadExperiment:
    def test_load_and_search(self, server_deps):
        sm, mock_client = server_deps
        from vast_mcp.mcp_server import _save_experiment, _load_experiment
        _save_experiment(sm, name="test-load", specs={"gpu_name": "RTX_4090", "num_gpus": 2}, summary="Test")
        mock_client.search_offers.return_value = [
            {"id": 10, "gpu_name": "RTX 4090", "num_gpus": 2, "gpu_ram": 24000,
             "cpu_ram": 64000, "disk_space": 200, "dph_total": 0.8, "reliability": 0.99}
        ]
        result = _load_experiment(sm, mock_client, name="test-load")
        assert "RTX 4090" in result
        mock_client.search_offers.assert_called_once()

    def test_load_with_overrides(self, server_deps):
        sm, mock_client = server_deps
        from vast_mcp.mcp_server import _save_experiment, _load_experiment
        _save_experiment(sm, name="override-test", specs={"gpu_name": "RTX_4090", "num_gpus": 2}, summary="Test")
        mock_client.search_offers.return_value = []
        _load_experiment(sm, mock_client, name="override-test", overrides={"num_gpus": 4})
        call_query = mock_client.search_offers.call_args[0][0]
        assert "num_gpus>=4" in call_query

    def test_load_nonexistent(self, server_deps):
        sm, mock_client = server_deps
        from vast_mcp.mcp_server import _load_experiment
        result = _load_experiment(sm, mock_client, name="nope")
        assert "not found" in result.lower()


class TestDeleteExperiment:
    def test_delete_existing(self, server_deps):
        sm, mock_client = server_deps
        from vast_mcp.mcp_server import _save_experiment, _delete_experiment
        _save_experiment(sm, name="del-me", specs={}, summary="Delete me")
        result = _delete_experiment(sm, name="del-me")
        assert "Deleted" in result or "deleted" in result
        assert "del-me" not in sm.load_experiments()

    def test_delete_nonexistent(self, server_deps):
        sm, mock_client = server_deps
        from vast_mcp.mcp_server import _delete_experiment
        result = _delete_experiment(sm, name="nope")
        assert "not found" in result.lower()
