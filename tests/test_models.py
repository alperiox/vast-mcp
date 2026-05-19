import json

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
