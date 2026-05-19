"""Integration test: save experiment -> load -> create -> register service -> check -> destroy."""

from unittest.mock import MagicMock, patch

from vast_mcp.models import Config
from vast_mcp.state import StateManager
from vast_mcp.mcp_server import (
    _save_experiment,
    _load_experiment,
    _create_instance,
    _register_service,
    _check_services,
    _destroy_instance,
    _list_instances,
)


def test_full_workflow(tmp_state_dir, monkeypatch):
    sm = StateManager(tmp_state_dir)
    sm.save_config(Config(api_key="test-key"))
    mock_client = MagicMock()

    # 1. Save experiment
    result = _save_experiment(
        sm, name="test-exp",
        specs={"gpu_name": "RTX_4090", "num_gpus": 2, "gpu_ram_min": 24000},
        summary="Integration test experiment",
        image="pytorch/pytorch:latest",
    )
    assert "Saved" in result

    # 2. Load experiment (searches offers)
    mock_client.search_offers.return_value = [
        {"id": 42, "gpu_name": "RTX 4090", "num_gpus": 2, "gpu_ram": 24000,
         "cpu_ram": 64000, "disk_space": 200, "dph_total": 0.8, "reliability": 0.99}
    ]
    result = _load_experiment(sm, mock_client, name="test-exp")
    assert "RTX 4090" in result

    # 3. Create instance from offer
    mock_client.create_instance.return_value = {"new_contract": 100}
    mock_client.show_instances.return_value = [
        {"id": 100, "machine_id": 50, "gpu_name": "RTX 4090", "num_gpus": 2,
         "image_uuid": "pytorch/pytorch:latest", "actual_status": "running"}
    ]
    result = _create_instance(sm, mock_client, offer_id=42, image="pytorch/pytorch:latest", disk_space=50)
    assert "100" in result

    # 4. Register service
    result = _register_service(sm, instance_id=100, name="jupyter", port=8888,
                               endpoint="http://1.2.3.4:8888", summary="Jupyter Lab")
    assert "Registered" in result

    # 5. Check services
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    monkeypatch.setattr("vast_mcp.mcp_server.httpx.get", lambda *a, **kw: mock_resp)
    result = _check_services(sm, instance_id=100)
    assert "up" in result.lower()

    # 6. List instances
    result = _list_instances(sm, mock_client)
    assert "RTX 4090" in result
    assert "jupyter" in result

    # 7. Destroy instance
    result = _destroy_instance(sm, mock_client, instance_id=100)
    assert "Destroyed" in result
    assert 100 not in sm.load_instances()
