# Vast.ai MCP Server — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Python MCP server for managing Vast.ai GPU instances with service tracking, experiment templates, and idle monitoring.

**Architecture:** One Python package (`vast_mcp`) with two entry points — a stdio MCP server and a cron monitor. State persisted as JSON in `~/.vast-mcp/`. FastMCP for the MCP layer, `vastai` SDK for Vast.ai API calls.

**Tech Stack:** Python 3.12+, FastMCP (mcp SDK), vastai SDK, uv, hatchling, pytest

---

### Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `src/vast_mcp/__init__.py`

- [ ] **Step 1: Create `pyproject.toml`**

```toml
[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[project]
name = "vast-mcp"
version = "0.1.0"
description = "MCP server for managing Vast.ai GPU instances"
requires-python = ">=3.12"
license = "MIT"
dependencies = [
    "mcp>=1.0",
    "vastai>=1.0",
    "httpx>=0.27",
]

[project.scripts]
vast-mcp = "vast_mcp.mcp_server:main"
vast-monitor = "vast_mcp.monitor:main"

[tool.hatch.build.targets.wheel]
packages = ["src/vast_mcp"]

[tool.pytest.ini_options]
testpaths = ["tests"]
pythonpath = ["src"]

[dependency-groups]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
]
```

- [ ] **Step 2: Create `src/vast_mcp/__init__.py`**

```python
"""Vast.ai MCP server for managing GPU instances."""
```

- [ ] **Step 3: Create test directory and conftest**

Create `tests/__init__.py` (empty) and `tests/conftest.py`:

```python
import json
import os
import tempfile
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def tmp_state_dir(tmp_path):
    """Provide a temporary state directory instead of ~/.vast-mcp/."""
    return tmp_path


@pytest.fixture
def mock_vast_client():
    """Provide a mock VastAI client."""
    return MagicMock()
```

- [ ] **Step 4: Initialize project with uv**

Run: `cd /Users/alperiox/Desktop/coding/vast-mcp && uv sync`
Expected: Dependencies installed, `.venv` created.

- [ ] **Step 5: Verify pytest runs**

Run: `cd /Users/alperiox/Desktop/coding/vast-mcp && uv run pytest --co -q`
Expected: "no tests ran" (no test files yet), exit 0 or 5.

- [ ] **Step 6: Commit**

```bash
git init
git add pyproject.toml src/ tests/
git commit -m "chore: scaffold project with pyproject.toml and package structure"
```

---

### Task 2: Data Models

**Files:**
- Create: `src/vast_mcp/models.py`
- Create: `tests/test_models.py`

- [ ] **Step 1: Write failing tests for data models**

Create `tests/test_models.py`:

```python
import json
from datetime import datetime, timezone

from vast_mcp.models import Config, Experiment, ExperimentSpecs, Instance, Service


class TestConfig:
    def test_defaults(self):
        config = Config()
        assert config.api_key is None
        assert config.idle_threshold_hours == 10
        assert config.monitor_interval_minutes == 30
        assert config.default_instance_type == "container"
        assert config.default_sort == "dph_total"
        assert config.default_max_results == 10

    def test_to_dict_roundtrip(self):
        config = Config(api_key="test-key", idle_threshold_hours=12)
        data = config.to_dict()
        restored = Config.from_dict(data)
        assert restored.api_key == "test-key"
        assert restored.idle_threshold_hours == 12

    def test_json_serializable(self):
        config = Config()
        serialized = json.dumps(config.to_dict())
        assert isinstance(serialized, str)


class TestService:
    def test_create(self):
        svc = Service(
            name="vllm",
            port=8000,
            endpoint="http://1.2.3.4:8000/v1/completions",
            summary="vLLM serving Llama-3",
        )
        assert svc.name == "vllm"
        assert svc.port == 8000
        assert svc.registered_at is not None
        assert svc.last_alive is None

    def test_to_dict_roundtrip(self):
        svc = Service(name="api", port=3000, summary="REST API")
        data = svc.to_dict()
        restored = Service.from_dict(data)
        assert restored.name == "api"
        assert restored.port == 3000
        assert restored.endpoint is None


class TestInstance:
    def test_create(self):
        inst = Instance(
            instance_id=12345,
            machine_id=6789,
            gpu_name="RTX 4090",
            num_gpus=2,
            image="pytorch/pytorch:latest",
            status="running",
        )
        assert inst.instance_id == 12345
        assert inst.services == []
        assert inst.experiment_name is None

    def test_to_dict_roundtrip(self):
        svc = Service(name="test", port=80, summary="test svc")
        inst = Instance(
            instance_id=1,
            machine_id=2,
            gpu_name="A100",
            num_gpus=1,
            image="ubuntu",
            status="running",
            services=[svc],
            experiment_name="exp1",
        )
        data = inst.to_dict()
        restored = Instance.from_dict(data)
        assert restored.instance_id == 1
        assert len(restored.services) == 1
        assert restored.services[0].name == "test"
        assert restored.experiment_name == "exp1"


class TestExperiment:
    def test_create(self):
        specs = ExperimentSpecs(gpu_name="RTX_4090", num_gpus=2)
        exp = Experiment(
            name="llama-ft",
            summary="Fine-tune Llama",
            specs=specs,
        )
        assert exp.name == "llama-ft"
        assert exp.specs.gpu_name == "RTX_4090"
        assert exp.image is None

    def test_to_query_string(self):
        specs = ExperimentSpecs(
            gpu_name="RTX_4090",
            num_gpus=2,
            gpu_ram_min=24000,
            max_dph=1.5,
        )
        query = specs.to_query_string()
        assert "gpu_name=RTX_4090" in query
        assert "num_gpus>=2" in query
        assert "gpu_ram>=24000" in query
        assert "dph_total<=1.5" in query

    def test_to_dict_roundtrip(self):
        specs = ExperimentSpecs(gpu_name="A100", num_gpus=4, disk_space_min=200)
        exp = Experiment(
            name="big-train",
            summary="Large training run",
            specs=specs,
            image="nvcr.io/nvidia/pytorch:24.01",
        )
        data = exp.to_dict()
        restored = Experiment.from_dict(data)
        assert restored.name == "big-train"
        assert restored.specs.num_gpus == 4
        assert restored.image == "nvcr.io/nvidia/pytorch:24.01"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_models.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vast_mcp.models'`

- [ ] **Step 3: Implement models**

Create `src/vast_mcp/models.py`:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Config:
    api_key: str | None = None
    idle_threshold_hours: int = 10
    monitor_interval_minutes: int = 30
    default_instance_type: str = "container"  # "container" | "vm" | "all"
    default_sort: str = "dph_total"
    default_max_results: int = 10

    def to_dict(self) -> dict:
        return {
            "api_key": self.api_key,
            "idle_threshold_hours": self.idle_threshold_hours,
            "monitor_interval_minutes": self.monitor_interval_minutes,
            "default_instance_type": self.default_instance_type,
            "default_sort": self.default_sort,
            "default_max_results": self.default_max_results,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Config:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Service:
    name: str
    port: int
    summary: str
    endpoint: str | None = None
    registered_at: str = field(default_factory=_now_iso)
    last_alive: str | None = None

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "port": self.port,
            "summary": self.summary,
            "endpoint": self.endpoint,
            "registered_at": self.registered_at,
            "last_alive": self.last_alive,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Service:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Instance:
    instance_id: int
    machine_id: int
    gpu_name: str
    num_gpus: int
    image: str
    status: str
    created_at: str = field(default_factory=_now_iso)
    last_checked: str = field(default_factory=_now_iso)
    services: list[Service] = field(default_factory=list)
    experiment_name: str | None = None

    def to_dict(self) -> dict:
        return {
            "instance_id": self.instance_id,
            "machine_id": self.machine_id,
            "gpu_name": self.gpu_name,
            "num_gpus": self.num_gpus,
            "image": self.image,
            "status": self.status,
            "created_at": self.created_at,
            "last_checked": self.last_checked,
            "services": [s.to_dict() for s in self.services],
            "experiment_name": self.experiment_name,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Instance:
        services = [Service.from_dict(s) for s in data.get("services", [])]
        filtered = {k: v for k, v in data.items() if k in cls.__dataclass_fields__ and k != "services"}
        return cls(services=services, **filtered)


@dataclass
class ExperimentSpecs:
    gpu_name: str | None = None
    num_gpus: int | None = None
    gpu_ram_min: int | None = None
    cpu_ram_min: int | None = None
    disk_space_min: int | None = None
    max_dph: float | None = None

    def to_query_string(self) -> str:
        parts = []
        if self.gpu_name:
            parts.append(f"gpu_name={self.gpu_name}")
        if self.num_gpus is not None:
            parts.append(f"num_gpus>={self.num_gpus}")
        if self.gpu_ram_min is not None:
            parts.append(f"gpu_ram>={self.gpu_ram_min}")
        if self.cpu_ram_min is not None:
            parts.append(f"cpu_ram>={self.cpu_ram_min}")
        if self.disk_space_min is not None:
            parts.append(f"disk_space>={self.disk_space_min}")
        if self.max_dph is not None:
            parts.append(f"dph_total<={self.max_dph}")
        return " ".join(parts)

    def to_dict(self) -> dict:
        return {k: v for k, v in {
            "gpu_name": self.gpu_name,
            "num_gpus": self.num_gpus,
            "gpu_ram_min": self.gpu_ram_min,
            "cpu_ram_min": self.cpu_ram_min,
            "disk_space_min": self.disk_space_min,
            "max_dph": self.max_dph,
        }.items() if v is not None}

    @classmethod
    def from_dict(cls, data: dict) -> ExperimentSpecs:
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class Experiment:
    name: str
    summary: str
    specs: ExperimentSpecs
    image: str | None = None
    created_at: str = field(default_factory=_now_iso)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "summary": self.summary,
            "specs": self.specs.to_dict(),
            "image": self.image,
            "created_at": self.created_at,
        }

    @classmethod
    def from_dict(cls, data: dict) -> Experiment:
        specs = ExperimentSpecs.from_dict(data.get("specs", {}))
        return cls(
            name=data["name"],
            summary=data["summary"],
            specs=specs,
            image=data.get("image"),
            created_at=data.get("created_at", _now_iso()),
        )
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_models.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vast_mcp/models.py tests/test_models.py
git commit -m "feat: add data models for config, instance, service, experiment"
```

---

### Task 3: State Layer

**Files:**
- Create: `src/vast_mcp/state.py`
- Create: `tests/test_state.py`

- [ ] **Step 1: Write failing tests for state management**

Create `tests/test_state.py`:

```python
import json

from vast_mcp.models import Config, Experiment, ExperimentSpecs, Instance, Service
from vast_mcp.state import StateManager


class TestStateManagerConfig:
    def test_load_default_config_when_no_file(self, tmp_state_dir):
        sm = StateManager(tmp_state_dir)
        config = sm.load_config()
        assert config.idle_threshold_hours == 10
        assert config.default_instance_type == "container"

    def test_save_and_load_config(self, tmp_state_dir):
        sm = StateManager(tmp_state_dir)
        config = Config(api_key="my-key", idle_threshold_hours=12)
        sm.save_config(config)
        loaded = sm.load_config()
        assert loaded.api_key == "my-key"
        assert loaded.idle_threshold_hours == 12

    def test_update_config_key(self, tmp_state_dir):
        sm = StateManager(tmp_state_dir)
        sm.save_config(Config())
        sm.update_config("idle_threshold_hours", 8)
        config = sm.load_config()
        assert config.idle_threshold_hours == 8

    def test_update_config_rejects_invalid_key(self, tmp_state_dir):
        sm = StateManager(tmp_state_dir)
        sm.save_config(Config())
        import pytest
        with pytest.raises(ValueError, match="Unknown config key"):
            sm.update_config("nonexistent_key", "value")


class TestStateManagerInstances:
    def test_load_empty_instances(self, tmp_state_dir):
        sm = StateManager(tmp_state_dir)
        instances = sm.load_instances()
        assert instances == {}

    def test_save_and_load_instance(self, tmp_state_dir):
        sm = StateManager(tmp_state_dir)
        inst = Instance(
            instance_id=123,
            machine_id=456,
            gpu_name="RTX 4090",
            num_gpus=1,
            image="pytorch",
            status="running",
        )
        sm.save_instance(inst)
        loaded = sm.load_instances()
        assert 123 in loaded
        assert loaded[123].gpu_name == "RTX 4090"

    def test_remove_instance(self, tmp_state_dir):
        sm = StateManager(tmp_state_dir)
        inst = Instance(
            instance_id=123,
            machine_id=456,
            gpu_name="A100",
            num_gpus=1,
            image="ubuntu",
            status="running",
        )
        sm.save_instance(inst)
        sm.remove_instance(123)
        loaded = sm.load_instances()
        assert 123 not in loaded

    def test_update_instance_status(self, tmp_state_dir):
        sm = StateManager(tmp_state_dir)
        inst = Instance(
            instance_id=1,
            machine_id=2,
            gpu_name="V100",
            num_gpus=1,
            image="img",
            status="running",
        )
        sm.save_instance(inst)
        sm.update_instance_status(1, "stopped")
        loaded = sm.load_instances()
        assert loaded[1].status == "stopped"


class TestStateManagerExperiments:
    def test_load_empty_experiments(self, tmp_state_dir):
        sm = StateManager(tmp_state_dir)
        experiments = sm.load_experiments()
        assert experiments == {}

    def test_save_and_load_experiment(self, tmp_state_dir):
        sm = StateManager(tmp_state_dir)
        exp = Experiment(
            name="test-exp",
            summary="A test experiment",
            specs=ExperimentSpecs(gpu_name="RTX_4090", num_gpus=2),
        )
        sm.save_experiment(exp)
        loaded = sm.load_experiments()
        assert "test-exp" in loaded
        assert loaded["test-exp"].specs.num_gpus == 2

    def test_delete_experiment(self, tmp_state_dir):
        sm = StateManager(tmp_state_dir)
        exp = Experiment(
            name="del-me",
            summary="To be deleted",
            specs=ExperimentSpecs(),
        )
        sm.save_experiment(exp)
        sm.delete_experiment("del-me")
        loaded = sm.load_experiments()
        assert "del-me" not in loaded

    def test_delete_nonexistent_raises(self, tmp_state_dir):
        sm = StateManager(tmp_state_dir)
        import pytest
        with pytest.raises(KeyError):
            sm.delete_experiment("nope")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_state.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vast_mcp.state'`

- [ ] **Step 3: Implement state layer**

Create `src/vast_mcp/state.py`:

```python
from __future__ import annotations

import json
from pathlib import Path

from vast_mcp.models import Config, Experiment, Instance


class StateManager:
    def __init__(self, state_dir: Path | str):
        self.state_dir = Path(state_dir)
        self.state_dir.mkdir(parents=True, exist_ok=True)

    def _read_json(self, filename: str) -> dict:
        path = self.state_dir / filename
        if not path.exists():
            return {}
        return json.loads(path.read_text())

    def _write_json(self, filename: str, data: dict) -> None:
        path = self.state_dir / filename
        path.write_text(json.dumps(data, indent=2))

    # --- Config ---

    def load_config(self) -> Config:
        data = self._read_json("config.json")
        if not data:
            return Config()
        return Config.from_dict(data)

    def save_config(self, config: Config) -> None:
        self._write_json("config.json", config.to_dict())

    def update_config(self, key: str, value) -> None:
        config = self.load_config()
        if key not in Config.__dataclass_fields__:
            raise ValueError(f"Unknown config key: {key}")
        setattr(config, key, value)
        self.save_config(config)

    # --- Instances ---

    def load_instances(self) -> dict[int, Instance]:
        data = self._read_json("instances.json")
        raw = data.get("instances", {})
        return {int(k): Instance.from_dict(v) for k, v in raw.items()}

    def save_instance(self, instance: Instance) -> None:
        instances = self.load_instances()
        instances[instance.instance_id] = instance
        self._write_instances(instances)

    def remove_instance(self, instance_id: int) -> None:
        instances = self.load_instances()
        instances.pop(instance_id, None)
        self._write_instances(instances)

    def update_instance_status(self, instance_id: int, status: str) -> None:
        instances = self.load_instances()
        if instance_id in instances:
            instances[instance_id].status = status
            self._write_instances(instances)

    def _write_instances(self, instances: dict[int, Instance]) -> None:
        data = {"instances": {str(k): v.to_dict() for k, v in instances.items()}}
        self._write_json("instances.json", data)

    # --- Experiments ---

    def load_experiments(self) -> dict[str, Experiment]:
        data = self._read_json("experiments.json")
        raw = data.get("experiments", {})
        return {k: Experiment.from_dict(v) for k, v in raw.items()}

    def save_experiment(self, experiment: Experiment) -> None:
        experiments = self.load_experiments()
        experiments[experiment.name] = experiment
        self._write_experiments(experiments)

    def delete_experiment(self, name: str) -> None:
        experiments = self.load_experiments()
        if name not in experiments:
            raise KeyError(f"Experiment not found: {name}")
        del experiments[name]
        self._write_experiments(experiments)

    def _write_experiments(self, experiments: dict[str, Experiment]) -> None:
        data = {"experiments": {k: v.to_dict() for k, v in experiments.items()}}
        self._write_json("experiments.json", data)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_state.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vast_mcp/state.py tests/test_state.py
git commit -m "feat: add state manager for JSON persistence"
```

---

### Task 4: Vast Client Wrapper

**Files:**
- Create: `src/vast_mcp/vast_client.py`
- Create: `tests/test_vast_client.py`

The wrapper keeps Vast.ai SDK details out of the MCP tools. It handles API key resolution and translates SDK responses into our models.

- [ ] **Step 1: Write failing tests for the client wrapper**

Create `tests/test_vast_client.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_vast_client.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'vast_mcp.vast_client'`

- [ ] **Step 3: Implement the client wrapper**

Create `src/vast_mcp/vast_client.py`:

```python
from __future__ import annotations

import os
from pathlib import Path

from vast_mcp.models import Config

VAST_CLI_KEY_PATH = Path.home() / ".config" / "vastai" / "vast_api_key"


class VastClient:
    def __init__(self, config: Config):
        self.api_key = self._resolve_api_key(config)
        from vastai import VastAI
        self._sdk = VastAI(api_key=self.api_key)
        self._config = config

    @staticmethod
    def _resolve_api_key(config: Config) -> str:
        if config.api_key:
            return config.api_key
        env_key = os.environ.get("VAST_API_KEY")
        if env_key:
            return env_key
        if VAST_CLI_KEY_PATH.exists():
            return VAST_CLI_KEY_PATH.read_text().strip()
        raise ValueError(
            "No Vast.ai API key found. Set it via config, VAST_API_KEY env var, "
            "or run 'vastai set api-key YOUR_KEY'."
        )

    def search_offers(
        self, query: str, sort_by: str | None = None, max_results: int | None = None
    ) -> list[dict]:
        instance_type = self._config.default_instance_type
        type_filter = ""
        if instance_type == "container":
            type_filter = "rentable=true machine_id!=0"
        elif instance_type == "vm":
            type_filter = "rentable=true is_vm=true"
        else:
            type_filter = "rentable=true"

        full_query = f"{type_filter} {query}".strip()
        sort = sort_by or self._config.default_sort
        limit = max_results or self._config.default_max_results

        result = self._sdk.search_offers(query=full_query, sort_order=sort, limit=limit)
        if result is None:
            return []
        return result if isinstance(result, list) else []

    def get_offer_details(self, offer_id: int) -> dict | None:
        results = self._sdk.search_offers(query=f"id={offer_id}")
        if results and isinstance(results, list) and len(results) > 0:
            return results[0]
        return None

    def create_instance(self, offer_id: int, image: str, disk_space: float, **kwargs) -> dict:
        return self._sdk.create_instance(
            id=offer_id,
            image=image,
            disk=disk_space,
            **kwargs,
        )

    def show_instances(self) -> list[dict]:
        result = self._sdk.show_instances()
        if result is None:
            return []
        return result if isinstance(result, list) else []

    def stop_instance(self, instance_id: int) -> None:
        self._sdk.stop_instance(id=instance_id)

    def start_instance(self, instance_id: int) -> None:
        self._sdk.start_instance(id=instance_id)

    def destroy_instance(self, instance_id: int) -> None:
        self._sdk.destroy_instance(id=instance_id)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_vast_client.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vast_mcp/vast_client.py tests/test_vast_client.py
git commit -m "feat: add Vast.ai client wrapper with API key resolution"
```

---

### Task 5: MCP Server — Discovery & Config Tools

**Files:**
- Create: `src/vast_mcp/mcp_server.py`
- Create: `tests/test_mcp_server.py`

- [ ] **Step 1: Write failing tests for discovery and config tools**

Create `tests/test_mcp_server.py`:

```python
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from vast_mcp.models import Config
from vast_mcp.state import StateManager


@pytest.fixture
def server_deps(tmp_state_dir):
    """Set up a state manager and mock vast client for testing MCP tool functions."""
    sm = StateManager(tmp_state_dir)
    sm.save_config(Config(api_key="test-key"))
    mock_client = MagicMock()
    return sm, mock_client


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_mcp_server.py -v`
Expected: FAIL — `cannot import name '_search_offers' from 'vast_mcp.mcp_server'`

- [ ] **Step 3: Implement MCP server with discovery and config tools**

Create `src/vast_mcp/mcp_server.py`:

```python
from __future__ import annotations

from pathlib import Path

from mcp.server.fastmcp import FastMCP

from vast_mcp.models import Config
from vast_mcp.state import StateManager
from vast_mcp.vast_client import VastClient

DEFAULT_STATE_DIR = Path.home() / ".vast-mcp"

mcp_app = FastMCP(name="vast-mcp")

# --- Globals initialized at startup ---
_state: StateManager | None = None
_client: VastClient | None = None


def _get_state() -> StateManager:
    global _state
    if _state is None:
        _state = StateManager(DEFAULT_STATE_DIR)
    return _state


def _get_client() -> VastClient:
    global _client
    if _client is None:
        config = _get_state().load_config()
        _client = VastClient(config)
    return _client


# === Internal functions (testable without MCP) ===

def _search_offers(
    sm: StateManager, client: VastClient, query: str,
    sort_by: str | None = None, max_results: int | None = None,
) -> str:
    offers = client.search_offers(query, sort_by, max_results)
    if not offers:
        return "No offers found matching the query."

    lines = []
    header = f"{'ID':>8} {'GPU':>16} {'#GPU':>5} {'VRAM':>8} {'RAM':>8} {'Disk':>8} {'$/hr':>8} {'Rel':>6}"
    lines.append(header)
    lines.append("-" * len(header))
    for o in offers:
        lines.append(
            f"{o.get('id', '?'):>8} "
            f"{o.get('gpu_name', '?'):>16} "
            f"{o.get('num_gpus', '?'):>5} "
            f"{o.get('gpu_ram', 0) / 1000:>7.1f}G "
            f"{o.get('cpu_ram', 0) / 1000:>7.1f}G "
            f"{o.get('disk_space', 0):>7.0f}G "
            f"{o.get('dph_total', 0):>8.3f} "
            f"{o.get('reliability', 0):>5.2f}"
        )
    return "\n".join(lines)


def _get_offer_details(client: VastClient, offer_id: int) -> str:
    offer = client.get_offer_details(offer_id)
    if offer is None:
        return f"Offer {offer_id} not found."

    fields = [
        f"Offer ID: {offer.get('id')}",
        f"GPU: {offer.get('gpu_name')} x{offer.get('num_gpus', 1)}",
        f"GPU RAM: {offer.get('gpu_ram', 0) / 1000:.1f} GB",
        f"CPU Cores: {offer.get('cpu_cores', '?')}",
        f"CPU RAM: {offer.get('cpu_ram', 0) / 1000:.1f} GB",
        f"Disk: {offer.get('disk_space', 0):.0f} GB",
        f"Price: ${offer.get('dph_total', 0):.3f}/hr",
        f"Reliability: {offer.get('reliability', 0):.2f}",
        f"Location: {offer.get('geolocation', '?')}",
        f"CUDA: {offer.get('cuda_max_good', '?')}",
        f"Ports: {offer.get('direct_port_count', '?')}",
        f"Net Down/Up: {offer.get('inet_down', '?')}/{offer.get('inet_up', '?')} Mbps",
    ]
    return "\n".join(fields)


def _get_config(sm: StateManager) -> str:
    config = sm.load_config()
    d = config.to_dict()
    # Mask API key
    if d.get("api_key"):
        d["api_key"] = d["api_key"][:4] + "..." + d["api_key"][-4:]
    lines = [f"{k}: {v}" for k, v in d.items()]
    return "\n".join(lines)


def _set_config(sm: StateManager, key: str, value) -> str:
    try:
        sm.update_config(key, value)
        return f"Updated {key} = {value}"
    except ValueError as e:
        return f"Error: {e}"


# === MCP Tool Definitions ===

@mcp_app.tool()
def search_offers(query: str, sort_by: str = "", max_results: int = 0) -> str:
    """Search available Vast.ai GPU machines.

    Args:
        query: Filter string (e.g., 'gpu_name=RTX_4090 num_gpus>=2 gpu_ram>=24000').
        sort_by: Sort field (default: dph_total). Options: dph_total, gpu_ram, num_gpus, dlperf.
        max_results: Max results to return (default: from config).
    """
    return _search_offers(
        _get_state(), _get_client(), query,
        sort_by=sort_by or None,
        max_results=max_results or None,
    )


@mcp_app.tool()
def get_offer_details(offer_id: int) -> str:
    """Get detailed information about a specific Vast.ai offer.

    Args:
        offer_id: The offer ID from search results.
    """
    return _get_offer_details(_get_client(), offer_id)


@mcp_app.tool()
def get_config() -> str:
    """Show current MCP server configuration (idle threshold, default filters, etc.)."""
    return _get_config(_get_state())


@mcp_app.tool()
def set_config(key: str, value: str) -> str:
    """Update a configuration value.

    Args:
        key: Config key (idle_threshold_hours, monitor_interval_minutes, default_instance_type, default_sort, default_max_results).
        value: New value. Numeric values will be converted automatically.
    """
    # Auto-convert numeric values
    converted: int | float | str = value
    try:
        converted = int(value)
    except ValueError:
        try:
            converted = float(value)
        except ValueError:
            pass
    return _set_config(_get_state(), key, converted)


def main():
    mcp_app.run(transport="stdio")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_mcp_server.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vast_mcp/mcp_server.py tests/test_mcp_server.py
git commit -m "feat: add MCP server with discovery and config tools"
```

---

### Task 6: MCP Server — Instance Lifecycle Tools

**Files:**
- Modify: `src/vast_mcp/mcp_server.py`
- Modify: `tests/test_mcp_server.py`

- [ ] **Step 1: Write failing tests for lifecycle tools**

Append to `tests/test_mcp_server.py`:

```python
from vast_mcp.models import Instance, Service


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
```

- [ ] **Step 2: Run tests to verify new tests fail**

Run: `uv run pytest tests/test_mcp_server.py -v -k "TestCreate or TestStop or TestStart or TestDestroy or TestList"`
Expected: FAIL — `cannot import name '_create_instance'`

- [ ] **Step 3: Add lifecycle internal functions and MCP tools to `mcp_server.py`**

Add these functions to `src/vast_mcp/mcp_server.py` (after the existing internal functions):

```python
def _create_instance(
    sm: StateManager, client: VastClient,
    offer_id: int, image: str, disk_space: float, **kwargs,
) -> str:
    try:
        result = client.create_instance(offer_id, image, disk_space, **kwargs)
        instance_id = result.get("new_contract")
        if not instance_id:
            return f"Instance creation response: {result}"

        # Fetch the new instance details from API
        api_instances = client.show_instances()
        for api_inst in api_instances:
            if api_inst.get("id") == instance_id:
                inst = Instance(
                    instance_id=instance_id,
                    machine_id=api_inst.get("machine_id", 0),
                    gpu_name=api_inst.get("gpu_name", "unknown"),
                    num_gpus=api_inst.get("num_gpus", 1),
                    image=api_inst.get("image_uuid", image),
                    status=api_inst.get("actual_status", "loading"),
                )
                sm.save_instance(inst)
                return f"Created instance {instance_id} ({inst.gpu_name} x{inst.num_gpus}). Status: {inst.status}"

        # If we can't find it yet, save with minimal info
        inst = Instance(
            instance_id=instance_id, machine_id=0,
            gpu_name="unknown", num_gpus=0, image=image, status="loading",
        )
        sm.save_instance(inst)
        return f"Created instance {instance_id}. Still loading — run list_instances to check status."
    except Exception as e:
        return f"Error creating instance: {e}"


def _stop_instance(sm: StateManager, client: VastClient, instance_id: int) -> str:
    instances = sm.load_instances()
    if instance_id not in instances:
        return f"Instance {instance_id} is not tracked. Use list_instances to see tracked instances."
    try:
        client.stop_instance(instance_id)
        sm.update_instance_status(instance_id, "stopped")
        return f"Stopped instance {instance_id}."
    except Exception as e:
        return f"Error stopping instance {instance_id}: {e}"


def _start_instance(sm: StateManager, client: VastClient, instance_id: int) -> str:
    instances = sm.load_instances()
    if instance_id not in instances:
        return f"Instance {instance_id} is not tracked."
    try:
        client.start_instance(instance_id)
        sm.update_instance_status(instance_id, "running")
        return f"Started instance {instance_id}."
    except Exception as e:
        return f"Error starting instance {instance_id}: {e}"


def _destroy_instance(sm: StateManager, client: VastClient, instance_id: int) -> str:
    instances = sm.load_instances()
    if instance_id not in instances:
        return f"Instance {instance_id} is not tracked."
    try:
        client.destroy_instance(instance_id)
        sm.remove_instance(instance_id)
        return f"Destroyed instance {instance_id} and removed from tracking."
    except Exception as e:
        return f"Error destroying instance {instance_id}: {e}"


def _list_instances(sm: StateManager, client: VastClient) -> str:
    instances = sm.load_instances()
    if not instances:
        return "No instances currently tracked."

    # Sync status from API
    try:
        api_instances = client.show_instances()
        api_status = {inst.get("id"): inst.get("actual_status", "unknown") for inst in api_instances}
        for iid, inst in instances.items():
            if iid in api_status:
                inst.status = api_status[iid]
                inst.last_checked = _now_iso()
        sm._write_instances(instances)
    except Exception:
        pass  # Use cached status if API fails

    lines = []
    for inst in instances.values():
        services_str = ""
        if inst.services:
            svc_names = [f"{s.name}:{s.port}" for s in inst.services]
            services_str = f" | Services: {', '.join(svc_names)}"
        exp_str = f" | Experiment: {inst.experiment_name}" if inst.experiment_name else ""
        lines.append(
            f"[{inst.instance_id}] {inst.gpu_name} x{inst.num_gpus} | "
            f"{inst.image} | {inst.status}{services_str}{exp_str}"
        )
    return "\n".join(lines)
```

Add `_now_iso` import at the top of the file:

```python
from vast_mcp.models import Config, Instance, Service, Experiment, ExperimentSpecs, _now_iso
```

And add the MCP tool decorators:

```python
@mcp_app.tool()
def create_instance(offer_id: int, image: str, disk_space: float = 50.0) -> str:
    """Create a new Vast.ai instance from a selected offer.

    Args:
        offer_id: The offer ID (from search_offers results).
        image: Docker image to use (e.g., 'pytorch/pytorch:latest').
        disk_space: Disk space in GB (default: 50).
    """
    return _create_instance(_get_state(), _get_client(), offer_id, image, disk_space)


@mcp_app.tool()
def stop_instance(instance_id: int) -> str:
    """Stop a running instance. Storage charges continue but compute stops.

    Args:
        instance_id: The instance ID to stop.
    """
    return _stop_instance(_get_state(), _get_client(), instance_id)


@mcp_app.tool()
def start_instance(instance_id: int) -> str:
    """Start a stopped instance.

    Args:
        instance_id: The instance ID to start.
    """
    return _start_instance(_get_state(), _get_client(), instance_id)


@mcp_app.tool()
def destroy_instance(instance_id: int) -> str:
    """Permanently destroy an instance and remove it from tracking.

    Args:
        instance_id: The instance ID to destroy.
    """
    return _destroy_instance(_get_state(), _get_client(), instance_id)


@mcp_app.tool()
def list_instances() -> str:
    """List all tracked instances with their status, services, and experiment info. Syncs status from Vast.ai API."""
    return _list_instances(_get_state(), _get_client())
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_mcp_server.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vast_mcp/mcp_server.py tests/test_mcp_server.py
git commit -m "feat: add instance lifecycle tools (create, stop, start, destroy, list)"
```

---

### Task 7: MCP Server — Service Registry Tools

**Files:**
- Modify: `src/vast_mcp/mcp_server.py`
- Modify: `tests/test_mcp_server.py`

- [ ] **Step 1: Write failing tests for service registry tools**

Append to `tests/test_mcp_server.py`:

```python
import httpx


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
        # Mock httpx.get to simulate endpoint check
        mock_response = MagicMock()
        mock_response.status_code = 200
        monkeypatch.setattr("vast_mcp.mcp_server.httpx.get", lambda *a, **kw: mock_response)
        from vast_mcp.mcp_server import _check_services
        result = _check_services(sm, instance_id=1)
        assert "alive" in result.lower() or "up" in result.lower()
```

- [ ] **Step 2: Run tests to verify new tests fail**

Run: `uv run pytest tests/test_mcp_server.py -v -k "TestRegister or TestUnregister or TestCheck"`
Expected: FAIL — `cannot import name '_register_service'`

- [ ] **Step 3: Add service registry internal functions and MCP tools**

Add to `src/vast_mcp/mcp_server.py`:

```python
import httpx


def _register_service(
    sm: StateManager, instance_id: int, name: str, port: int,
    summary: str, endpoint: str | None = None,
) -> str:
    instances = sm.load_instances()
    if instance_id not in instances:
        return f"Instance {instance_id} is not tracked."
    inst = instances[instance_id]
    # Replace if duplicate name
    inst.services = [s for s in inst.services if s.name != name]
    svc = Service(name=name, port=port, endpoint=endpoint, summary=summary)
    inst.services.append(svc)
    sm.save_instance(inst)
    return f"Registered service '{name}' on instance {instance_id} (port {port})."


def _unregister_service(sm: StateManager, instance_id: int, service_name: str) -> str:
    instances = sm.load_instances()
    if instance_id not in instances:
        return f"Instance {instance_id} is not tracked."
    inst = instances[instance_id]
    original_count = len(inst.services)
    inst.services = [s for s in inst.services if s.name != service_name]
    if len(inst.services) == original_count:
        return f"Service '{service_name}' not found on instance {instance_id}."
    sm.save_instance(inst)
    return f"Removed service '{service_name}' from instance {instance_id}."


def _check_services(sm: StateManager, instance_id: int | None = None) -> str:
    instances = sm.load_instances()
    if instance_id is not None:
        if instance_id not in instances:
            return f"Instance {instance_id} is not tracked."
        targets = {instance_id: instances[instance_id]}
    else:
        targets = instances

    all_services = []
    for iid, inst in targets.items():
        for svc in inst.services:
            all_services.append((iid, inst, svc))

    if not all_services:
        return "No services registered on any tracked instance."

    lines = []
    for iid, inst, svc in all_services:
        status = _probe_service(svc)
        if status == "up":
            svc.last_alive = _now_iso()
        lines.append(f"[{iid}] {svc.name}:{svc.port} — {status} ({svc.summary})")

    # Save updated last_alive timestamps
    for iid, inst, _ in all_services:
        sm.save_instance(inst)

    return "\n".join(lines)


def _probe_service(svc: Service) -> str:
    if svc.endpoint:
        try:
            resp = httpx.get(svc.endpoint, timeout=5)
            return f"up (HTTP {resp.status_code})"
        except Exception:
            return "down"
    # TCP connect fallback for non-HTTP services
    import socket
    try:
        # We'd need the IP — for now, just report no endpoint
        return "no endpoint configured"
    except Exception:
        return "down"
```

And the MCP tool decorators:

```python
@mcp_app.tool()
def register_service(
    instance_id: int, name: str, port: int, summary: str, endpoint: str = "",
) -> str:
    """Register a service running on an instance for liveness tracking.

    Args:
        instance_id: The instance ID.
        name: Service name (e.g., 'vllm', 'jupyter', 'api').
        port: Port number the service runs on.
        summary: Short description of the service.
        endpoint: Optional HTTP endpoint for liveness checks (e.g., 'http://x.x.x.x:8000/health').
    """
    return _register_service(
        _get_state(), instance_id, name, port, summary,
        endpoint=endpoint or None,
    )


@mcp_app.tool()
def unregister_service(instance_id: int, service_name: str) -> str:
    """Remove a service annotation from an instance.

    Args:
        instance_id: The instance ID.
        service_name: Name of the service to remove.
    """
    return _unregister_service(_get_state(), instance_id, service_name)


@mcp_app.tool()
def check_services(instance_id: int = 0) -> str:
    """Check liveness of registered service endpoints.

    Args:
        instance_id: Check a specific instance (0 = check all instances).
    """
    return _check_services(_get_state(), instance_id=instance_id or None)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_mcp_server.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vast_mcp/mcp_server.py tests/test_mcp_server.py
git commit -m "feat: add service registry tools (register, unregister, check)"
```

---

### Task 8: MCP Server — Experiment Template Tools

**Files:**
- Modify: `src/vast_mcp/mcp_server.py`
- Modify: `tests/test_mcp_server.py`

- [ ] **Step 1: Write failing tests for experiment tools**

Append to `tests/test_mcp_server.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify new tests fail**

Run: `uv run pytest tests/test_mcp_server.py -v -k "TestSave or TestListExp or TestLoad or TestDelete"`
Expected: FAIL — `cannot import name '_save_experiment'`

- [ ] **Step 3: Add experiment internal functions and MCP tools**

Add to `src/vast_mcp/mcp_server.py`:

```python
import json as json_module


def _save_experiment(
    sm: StateManager, name: str, specs: dict, summary: str, image: str | None = None,
) -> str:
    exp_specs = ExperimentSpecs.from_dict(specs)
    experiment = Experiment(name=name, summary=summary, specs=exp_specs, image=image)
    sm.save_experiment(experiment)
    return f"Saved experiment '{name}': {summary}"


def _list_experiments(sm: StateManager) -> str:
    experiments = sm.load_experiments()
    if not experiments:
        return "No experiments saved."
    lines = []
    for exp in experiments.values():
        specs_str = ", ".join(f"{k}={v}" for k, v in exp.specs.to_dict().items())
        image_str = f" | Image: {exp.image}" if exp.image else ""
        lines.append(f"  {exp.name}: {exp.summary} [{specs_str}]{image_str}")
    return "Saved experiments:\n" + "\n".join(lines)


def _load_experiment(
    sm: StateManager, client: VastClient, name: str, overrides: dict | None = None,
) -> str:
    experiments = sm.load_experiments()
    if name not in experiments:
        return f"Experiment '{name}' not found."
    exp = experiments[name]
    specs_dict = exp.specs.to_dict()
    if overrides:
        specs_dict.update(overrides)
    merged_specs = ExperimentSpecs.from_dict(specs_dict)
    query = merged_specs.to_query_string()
    offers = client.search_offers(query)
    if not offers:
        return f"Loaded experiment '{name}' but no matching offers found for query: {query}"
    result = _search_offers(sm, client, query)
    return f"Experiment '{name}': {exp.summary}\nQuery: {query}\n\n{result}"


def _delete_experiment(sm: StateManager, name: str) -> str:
    try:
        sm.delete_experiment(name)
        return f"Deleted experiment '{name}'."
    except KeyError:
        return f"Experiment '{name}' not found."
```

And MCP tool decorators:

```python
@mcp_app.tool()
def save_experiment(name: str, specs: str, summary: str, image: str = "") -> str:
    """Save a machine spec template for reuse.

    Args:
        name: Template name (e.g., 'llama-finetune').
        specs: JSON string of specs (e.g., '{"gpu_name": "RTX_4090", "num_gpus": 2, "gpu_ram_min": 24000}').
               Valid keys: gpu_name, num_gpus, gpu_ram_min, cpu_ram_min, disk_space_min, max_dph.
        summary: Short description of the experiment.
        image: Optional Docker image (e.g., 'pytorch/pytorch:2.3-cuda12.1').
    """
    try:
        specs_dict = json_module.loads(specs)
    except json_module.JSONDecodeError:
        return "Error: specs must be a valid JSON string."
    return _save_experiment(_get_state(), name, specs_dict, summary, image=image or None)


@mcp_app.tool()
def list_experiments() -> str:
    """List all saved experiment templates."""
    return _list_experiments(_get_state())


@mcp_app.tool()
def load_experiment(name: str, overrides: str = "") -> str:
    """Load an experiment template and search for matching offers.

    Args:
        name: Template name to load.
        overrides: Optional JSON string of spec overrides (e.g., '{"num_gpus": 4}').
    """
    overrides_dict = None
    if overrides:
        try:
            overrides_dict = json_module.loads(overrides)
        except json_module.JSONDecodeError:
            return "Error: overrides must be a valid JSON string."
    return _load_experiment(_get_state(), _get_client(), name, overrides_dict)


@mcp_app.tool()
def delete_experiment(name: str) -> str:
    """Delete a saved experiment template.

    Args:
        name: Template name to delete.
    """
    return _delete_experiment(_get_state(), name)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_mcp_server.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vast_mcp/mcp_server.py tests/test_mcp_server.py
git commit -m "feat: add experiment template tools (save, list, load, delete)"
```

---

### Task 9: Notifications Module

**Files:**
- Create: `src/vast_mcp/notifications.py`
- Create: `tests/test_notifications.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_notifications.py`:

```python
from unittest.mock import patch, MagicMock

from vast_mcp.notifications import notify


class TestNotify:
    def test_calls_osascript(self):
        with patch("vast_mcp.notifications.subprocess.run") as mock_run:
            mock_run.return_value = MagicMock(returncode=0)
            notify("Test Title", "Test message body")
            mock_run.assert_called_once()
            cmd = mock_run.call_args[0][0]
            assert "osascript" in cmd[0]
            assert "Test Title" in cmd[1]
            assert "Test message body" in cmd[1]

    def test_handles_osascript_failure(self):
        with patch("vast_mcp.notifications.subprocess.run", side_effect=FileNotFoundError):
            # Should not raise
            notify("Title", "Body")
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_notifications.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement notifications**

Create `src/vast_mcp/notifications.py`:

```python
from __future__ import annotations

import subprocess


def notify(title: str, message: str) -> None:
    script = f'display notification "{message}" with title "{title}"'
    try:
        subprocess.run(["osascript", "-e", script], capture_output=True, timeout=10)
    except Exception:
        pass  # Notification is best-effort
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_notifications.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vast_mcp/notifications.py tests/test_notifications.py
git commit -m "feat: add macOS notification helper"
```

---

### Task 10: Monitor

**Files:**
- Create: `src/vast_mcp/monitor.py`
- Create: `tests/test_monitor.py`

- [ ] **Step 1: Write failing tests for monitor logic**

Create `tests/test_monitor.py`:

```python
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

from vast_mcp.models import Config, Instance, Service
from vast_mcp.monitor import Monitor
from vast_mcp.state import StateManager


@pytest.fixture
def monitor_deps(tmp_state_dir):
    sm = StateManager(tmp_state_dir)
    sm.save_config(Config(api_key="test-key", idle_threshold_hours=10))
    mock_client = MagicMock()
    return sm, mock_client


import pytest


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/test_monitor.py -v`
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement monitor**

Create `src/vast_mcp/monitor.py`:

```python
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

from vast_mcp.models import Config, Instance, Service, _now_iso
from vast_mcp.notifications import notify
from vast_mcp.state import StateManager
from vast_mcp.vast_client import VastClient

DEFAULT_STATE_DIR = Path.home() / ".vast-mcp"

logger = logging.getLogger("vast-monitor")


class Monitor:
    def __init__(self, state: StateManager, client: VastClient):
        self.state = state
        self.client = client
        self.config = state.load_config()

    def sync_status(self) -> list[int]:
        """Sync instance statuses from Vast.ai API. Returns list of removed instance IDs."""
        instances = self.state.load_instances()
        if not instances:
            return []

        api_instances = self.client.show_instances()
        api_ids = {inst.get("id") for inst in api_instances}
        api_status = {inst.get("id"): inst.get("actual_status", "unknown") for inst in api_instances}

        removed = []
        for iid in list(instances.keys()):
            if iid not in api_ids:
                notify(
                    "Vast.ai Monitor",
                    f"Instance {iid} no longer exists on Vast.ai. Removed from tracking.",
                )
                self.state.remove_instance(iid)
                removed.append(iid)
            else:
                instances[iid].status = api_status.get(iid, "unknown")
                instances[iid].last_checked = _now_iso()
                self.state.save_instance(instances[iid])

        return removed

    def check_liveness(self) -> None:
        """Check liveness of all registered service endpoints."""
        instances = self.state.load_instances()
        for inst in instances.values():
            for svc in inst.services:
                if svc.endpoint:
                    try:
                        resp = httpx.get(svc.endpoint, timeout=5)
                        if resp.status_code < 500:
                            svc.last_alive = _now_iso()
                    except Exception:
                        pass
            self.state.save_instance(inst)

    def evaluate_idle(self) -> None:
        """Check for idle instances and send notifications."""
        instances = self.state.load_instances()
        threshold_hours = self.config.idle_threshold_hours
        now = datetime.now(timezone.utc)

        for inst in instances.values():
            if inst.status != "running" or not inst.services:
                continue

            all_idle = True
            max_idle_hours = 0.0
            for svc in inst.services:
                if svc.last_alive is None:
                    # Never been alive — skip (newly registered)
                    all_idle = False
                    continue
                last = datetime.fromisoformat(svc.last_alive)
                hours_idle = (now - last).total_seconds() / 3600
                max_idle_hours = max(max_idle_hours, hours_idle)
                if hours_idle < threshold_hours:
                    all_idle = False

            if all_idle and max_idle_hours >= threshold_hours:
                notify(
                    "Vast.ai Monitor",
                    f"Instance {inst.instance_id} ({inst.num_gpus}x {inst.gpu_name}) — "
                    f"all services down for {max_idle_hours:.0f}h. Still in use?",
                )

    def run(self) -> None:
        """Execute a full monitor cycle."""
        logger.info("Starting monitor run")
        self.sync_status()
        self.check_liveness()
        self.evaluate_idle()
        logger.info("Monitor run complete")


def _setup_logging(state_dir: Path) -> None:
    log_path = state_dir / "monitor.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler(sys.stderr),
        ],
    )


def _install(state_dir: Path) -> None:
    """Install a launchd plist for periodic monitoring."""
    config = StateManager(state_dir).load_config()
    interval = config.monitor_interval_minutes * 60
    plist_name = "com.vast-mcp.monitor"
    plist_dir = Path.home() / "Library" / "LaunchAgents"
    plist_dir.mkdir(parents=True, exist_ok=True)
    plist_path = plist_dir / f"{plist_name}.plist"

    # Find the vast-monitor executable
    import shutil
    vast_monitor_path = shutil.which("vast-monitor")
    if not vast_monitor_path:
        print("Error: vast-monitor not found in PATH. Install the package first.")
        sys.exit(1)

    plist_content = f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{plist_name}</string>
    <key>ProgramArguments</key>
    <array>
        <string>{vast_monitor_path}</string>
        <string>run</string>
    </array>
    <key>StartInterval</key>
    <integer>{interval}</integer>
    <key>StandardOutPath</key>
    <string>{state_dir / "monitor-stdout.log"}</string>
    <key>StandardErrorPath</key>
    <string>{state_dir / "monitor-stderr.log"}</string>
</dict>
</plist>"""

    plist_path.write_text(plist_content)
    import subprocess
    subprocess.run(["launchctl", "load", str(plist_path)], check=True)
    print(f"Installed and loaded {plist_path}")
    print(f"Monitor will run every {config.monitor_interval_minutes} minutes.")


def _uninstall() -> None:
    """Remove the launchd plist."""
    plist_name = "com.vast-mcp.monitor"
    plist_path = Path.home() / "Library" / "LaunchAgents" / f"{plist_name}.plist"
    if plist_path.exists():
        import subprocess
        subprocess.run(["launchctl", "unload", str(plist_path)], check=False)
        plist_path.unlink()
        print(f"Uninstalled {plist_path}")
    else:
        print("Monitor is not installed.")


def main():
    parser = argparse.ArgumentParser(description="Vast.ai instance monitor")
    parser.add_argument("command", choices=["run", "install", "uninstall"], help="Command to execute")
    args = parser.parse_args()

    state_dir = DEFAULT_STATE_DIR
    state_dir.mkdir(parents=True, exist_ok=True)

    if args.command == "install":
        _install(state_dir)
    elif args.command == "uninstall":
        _uninstall()
    elif args.command == "run":
        _setup_logging(state_dir)
        sm = StateManager(state_dir)
        config = sm.load_config()
        try:
            client = VastClient(config)
        except ValueError as e:
            logger.error(f"Cannot start monitor: {e}")
            sys.exit(1)
        monitor = Monitor(sm, client)
        monitor.run()


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/test_monitor.py -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add src/vast_mcp/monitor.py tests/test_monitor.py
git commit -m "feat: add monitor with liveness checks, idle detection, and launchd install"
```

---

### Task 11: Skill Files

**Files:**
- Create: `skills/provision-instance.md`
- Create: `skills/save-experiment.md`

- [ ] **Step 1: Create provision-instance skill**

Create `skills/provision-instance.md`:

```markdown
---
name: provision-instance
description: Use when the user wants to provision, rent, or spin up a Vast.ai GPU instance. Guides through spec definition, offer search, selection, and instance creation.
---

# Provision a Vast.ai Instance

Guide the user through provisioning a GPU instance step by step.

## Steps

1. **Understand requirements.** Ask the user what they need. Check if they want to use a saved experiment template:
   - Call `list_experiments` to show available templates
   - If they reference a template, call `load_experiment` (with optional overrides)
   - If they describe requirements loosely, extract specs: GPU type, GPU count, VRAM, RAM, disk, max price

2. **Search for offers.** Build a query string from the specs and call `search_offers`. Present the results as a table. Key fields to highlight: ID, GPU, count, VRAM, RAM, disk, $/hr, reliability.

3. **Let the user pick.** Never auto-select. Present the options and ask the user which offer they want. If none are suitable, refine the search with adjusted filters.

4. **Create the instance.** Once the user picks an offer ID:
   - Ask what Docker image to use (suggest common ones: `pytorch/pytorch:latest`, `nvidia/cuda:12.4.0-devel-ubuntu22.04`)
   - Ask disk space needed (default 50 GB)
   - Call `create_instance` with the chosen offer ID, image, and disk

5. **Register services.** After creation, ask the user:
   - "What will you run on this instance?"
   - Once they describe it, call `register_service` with name, port, endpoint (if HTTP), and summary
   - This enables liveness tracking by the monitor

6. **Offer to save as template.** Ask: "Want to save these specs as an experiment template for next time?" If yes, invoke the `save-experiment` skill or call `save_experiment` directly.
```

- [ ] **Step 2: Create save-experiment skill**

Create `skills/save-experiment.md`:

```markdown
---
name: save-experiment
description: Use when the user wants to save machine specs or experiment configuration for reuse. Extracts requirements from conversation context and saves as a named template.
---

# Save Experiment Configuration

Extract machine requirements from the conversation and save as a reusable template.

## Steps

1. **Review conversation context.** Look at what the user has discussed — GPU requirements, model sizes, training frameworks, inference needs. Extract:
   - `gpu_name`: Specific GPU model (e.g., RTX_4090, A100, H100)
   - `num_gpus`: Number of GPUs needed
   - `gpu_ram_min`: Minimum VRAM per GPU in MB (e.g., 24000 for 24 GB)
   - `cpu_ram_min`: Minimum system RAM in MB
   - `disk_space_min`: Minimum disk in GB
   - `max_dph`: Maximum price per hour in dollars

2. **Generate a name and summary.** Create a short, descriptive name (kebab-case, e.g., `llama3-finetune`, `sd-inference`). Write a 1-sentence summary of the experiment.

3. **Confirm with the user.** Present the extracted specs and ask for confirmation:
   > "I'll save this experiment as `{name}`:
   > - Summary: {summary}
   > - GPU: {gpu_name} x{num_gpus}
   > - VRAM: {gpu_ram_min} MB
   > - RAM: {cpu_ram_min} MB
   > - Disk: {disk_space_min} GB
   > - Max price: ${max_dph}/hr
   > - Image: {image or 'not set'}
   >
   > Does this look right?"

4. **Save.** Call `save_experiment` with the confirmed specs as a JSON string.

5. **Confirm saved.** Tell the user the template is saved and they can load it later with `load_experiment`.
```

- [ ] **Step 3: Commit**

```bash
git add skills/
git commit -m "feat: add provision-instance and save-experiment skill files"
```

---

### Task 12: Integration Test & Final Wiring

**Files:**
- Create: `tests/test_integration.py`
- Verify: all entry points work

- [ ] **Step 1: Write an integration test for the full workflow**

Create `tests/test_integration.py`:

```python
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
```

- [ ] **Step 2: Run the integration test**

Run: `uv run pytest tests/test_integration.py -v`
Expected: PASS

- [ ] **Step 3: Verify entry points are wired correctly**

Run: `uv run vast-mcp --help 2>&1 || echo "stdio server — no --help expected"`
Run: `uv run vast-monitor --help`
Expected: vast-monitor shows help with `run`, `install`, `uninstall` commands.

- [ ] **Step 4: Run full test suite**

Run: `uv run pytest -v`
Expected: All tests PASS.

- [ ] **Step 5: Commit**

```bash
git add tests/test_integration.py
git commit -m "test: add integration test for full workflow"
```

---

## File Summary

| File | Purpose |
|------|---------|
| `pyproject.toml` | Package config, deps, entry points |
| `src/vast_mcp/__init__.py` | Package init |
| `src/vast_mcp/models.py` | Dataclasses: Config, Service, Instance, ExperimentSpecs, Experiment |
| `src/vast_mcp/state.py` | JSON state persistence (read/write to `~/.vast-mcp/`) |
| `src/vast_mcp/vast_client.py` | Thin wrapper around `vastai` SDK with API key resolution |
| `src/vast_mcp/mcp_server.py` | FastMCP server with 15 tools + `main()` entry point |
| `src/vast_mcp/notifications.py` | macOS notification via `osascript` |
| `src/vast_mcp/monitor.py` | Cron monitor + launchd install/uninstall + `main()` entry point |
| `skills/provision-instance.md` | Skill: guided instance provisioning |
| `skills/save-experiment.md` | Skill: extract & save experiment from conversation |
| `tests/conftest.py` | Shared fixtures (tmp_state_dir, mock_vast_client) |
| `tests/test_models.py` | Model serialization tests |
| `tests/test_state.py` | State persistence tests |
| `tests/test_vast_client.py` | Client wrapper tests (mocked SDK) |
| `tests/test_mcp_server.py` | MCP tool function tests |
| `tests/test_notifications.py` | Notification tests |
| `tests/test_monitor.py` | Monitor logic tests |
| `tests/test_integration.py` | End-to-end workflow test |
