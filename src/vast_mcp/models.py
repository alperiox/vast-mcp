from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass
class Config:
    api_key: str | None = None
    ssh_key_path: str | None = None
    idle_threshold_hours: int = 10
    monitor_interval_minutes: int = 30
    default_instance_type: str = "container"  # "container" | "vm" | "all"
    default_sort: str = "dph_total"
    default_max_results: int = 10

    def to_dict(self) -> dict:
        return {
            "api_key": self.api_key,
            "ssh_key_path": self.ssh_key_path,
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
            parts.append(f"gpu_name={self.gpu_name.replace(' ', '_')}")
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
