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
