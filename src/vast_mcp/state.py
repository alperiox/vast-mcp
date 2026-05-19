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
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(data, indent=2))
        tmp.replace(path)  # atomic on POSIX

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
