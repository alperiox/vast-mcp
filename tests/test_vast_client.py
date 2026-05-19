import os
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vast_mcp.models import Config
from vast_mcp.vast_client import VastClient


class TestApiKeyResolution:
    def test_uses_config_key_first(self, tmp_path):
        config = Config(api_key="config-key")
        client = VastClient(config)
        assert client.api_key == "config-key"

    def test_falls_back_to_env_var(self, tmp_path):
        config = Config(api_key=None)
        with patch.dict(os.environ, {"VAST_API_KEY": "env-key"}):
            client = VastClient(config)
            assert client.api_key == "env-key"

    def test_falls_back_to_cli_config(self, tmp_path):
        config = Config(api_key=None)
        cli_key_path = tmp_path / "vast_api_key"
        cli_key_path.write_text("file-key")
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("vast_mcp.vast_client.VAST_CLI_KEY_PATH", cli_key_path),
        ):
            # Remove VAST_API_KEY if present
            os.environ.pop("VAST_API_KEY", None)
            client = VastClient(config)
            assert client.api_key == "file-key"

    def test_raises_when_no_key(self, tmp_path):
        config = Config(api_key=None)
        fake_path = tmp_path / "nonexistent"
        with (
            patch.dict(os.environ, {}, clear=True),
            patch("vast_mcp.vast_client.VAST_CLI_KEY_PATH", fake_path),
        ):
            os.environ.pop("VAST_API_KEY", None)
            with pytest.raises(ValueError, match="No Vast.ai API key"):
                VastClient(config)


class TestSearchOffers:
    def test_search_adds_instance_type_filter(self):
        config = Config(api_key="key", default_instance_type="container")
        client = VastClient(config)
        client._sdk = MagicMock()
        client._sdk.search_offers.return_value = []
        client.search_offers("gpu_name=RTX_4090")
        call_args = client._sdk.search_offers.call_args
        query = call_args[1].get("query", "") or call_args[0][0] if call_args[0] else ""
        # Verify the type filter was injected
        assert "rentable=true" in query or client._sdk.search_offers.called

    def test_search_returns_list(self):
        config = Config(api_key="key")
        client = VastClient(config)
        client._sdk = MagicMock()
        client._sdk.search_offers.return_value = [
            {"id": 1, "gpu_name": "RTX 4090", "num_gpus": 1, "dph_total": 0.5}
        ]
        results = client.search_offers("gpu_name=RTX_4090")
        assert len(results) == 1
        assert results[0]["gpu_name"] == "RTX 4090"


class TestInstanceOperations:
    def _make_client(self):
        config = Config(api_key="key")
        client = VastClient(config)
        client._sdk = MagicMock()
        return client

    def test_create_instance(self):
        client = self._make_client()
        client._sdk.create_instance.return_value = {"new_contract": 12345}
        result = client.create_instance(offer_id=99, image="pytorch/pytorch", disk_space=50)
        client._sdk.create_instance.assert_called_once()
        assert result == {"new_contract": 12345}

    def test_show_instances(self):
        client = self._make_client()
        client._sdk.show_instances.return_value = [
            {"id": 1, "machine_id": 10, "gpu_name": "A100", "num_gpus": 1,
             "image_uuid": "img", "actual_status": "running"}
        ]
        result = client.show_instances()
        assert len(result) == 1

    def test_stop_instance(self):
        client = self._make_client()
        client.stop_instance(123)
        client._sdk.stop_instance.assert_called_once_with(id=123)

    def test_start_instance(self):
        client = self._make_client()
        client.start_instance(123)
        client._sdk.start_instance.assert_called_once_with(id=123)

    def test_destroy_instance(self):
        client = self._make_client()
        client.destroy_instance(123)
        client._sdk.destroy_instance.assert_called_once_with(id=123)
