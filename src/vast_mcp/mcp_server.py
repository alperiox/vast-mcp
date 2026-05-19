from __future__ import annotations

import json as json_module
from pathlib import Path

import httpx
from mcp.server.fastmcp import FastMCP

from vast_mcp.models import Experiment, ExperimentSpecs, Instance, Service, _now_iso
from vast_mcp.state import StateManager
from vast_mcp.vast_client import VastClient

DEFAULT_STATE_DIR = Path.home() / ".vast-mcp"

mcp_app = FastMCP(name="vast-mcp")

# --- Globals initialized at startup ---
_state: StateManager | None = None


def _get_state() -> StateManager:
    global _state
    if _state is None:
        _state = StateManager(DEFAULT_STATE_DIR)
    return _state


def _get_client() -> VastClient:
    config = _get_state().load_config()
    return VastClient(config)


# ---------------------------------------------------------------------------
# Task 5: Discovery & Config internal functions
# ---------------------------------------------------------------------------


def _search_offers(sm: StateManager, client, query: str, sort_by: str | None = None, max_results: int | None = None) -> str:
    """Search for GPU offers and format as a table."""
    offers = client.search_offers(query, sort_by=sort_by, max_results=max_results)
    if not offers:
        return "No offers found matching the query."

    lines = ["GPU Offers:"]
    lines.append(f"{'ID':<8} {'GPU':<20} {'#GPU':<6} {'VRAM(MB)':<10} {'RAM(MB)':<10} {'Disk(GB)':<10} {'$/hr':<8} {'Reliability'}")
    lines.append("-" * 90)
    for o in offers:
        lines.append(
            f"{o.get('id', ''):<8} "
            f"{o.get('gpu_name', ''):<20} "
            f"{o.get('num_gpus', ''):<6} "
            f"{o.get('gpu_ram', ''):<10} "
            f"{o.get('cpu_ram', ''):<10} "
            f"{o.get('disk_space', ''):<10} "
            f"{o.get('dph_total', ''):<8} "
            f"{o.get('reliability', '')}"
        )
    return "\n".join(lines)


def _get_offer_details(client, offer_id: int) -> str:
    """Get detailed info for a single offer."""
    offer = client.get_offer_details(offer_id)
    if offer is None:
        return f"Offer {offer_id} not found."

    lines = [f"Offer Details (ID: {offer_id}):"]
    fields = [
        ("GPU", offer.get("gpu_name")),
        ("Num GPUs", offer.get("num_gpus")),
        ("GPU RAM (MB)", offer.get("gpu_ram")),
        ("CPU Cores", offer.get("cpu_cores")),
        ("CPU RAM (MB)", offer.get("cpu_ram")),
        ("Disk Space (GB)", offer.get("disk_space")),
        ("Price ($/hr)", offer.get("dph_total")),
        ("Reliability", offer.get("reliability")),
        ("Location", offer.get("geolocation")),
        ("CUDA Max", offer.get("cuda_max_good")),
        ("Direct Ports", offer.get("direct_port_count")),
        ("Download (Mbps)", offer.get("inet_down")),
        ("Upload (Mbps)", offer.get("inet_up")),
    ]
    for label, val in fields:
        if val is not None:
            lines.append(f"  {label}: {val}")
    return "\n".join(lines)


def _get_config(sm: StateManager) -> str:
    """Show current configuration, masking the API key."""
    config = sm.load_config()
    d = config.to_dict()
    if d.get("api_key"):
        key = d["api_key"]
        d["api_key"] = key[:4] + "****" + key[-4:] if len(key) > 8 else "****"
    lines = ["Current Configuration:"]
    for k, v in d.items():
        lines.append(f"  {k}: {v}")
    return "\n".join(lines)


def _set_config(sm: StateManager, key: str, value) -> str:
    """Update a single config field."""
    try:
        sm.update_config(key, value)
        return f"Updated {key} = {value}"
    except ValueError as e:
        return f"Error: {e}"


# ---------------------------------------------------------------------------
# Task 5: MCP tool wrappers — Discovery & Config
# ---------------------------------------------------------------------------


@mcp_app.tool()
def search_offers(query: str, sort_by: str | None = None, max_results: int | None = None) -> str:
    """Search for available GPU offers on Vast.ai."""
    return _search_offers(_get_state(), _get_client(), query=query, sort_by=sort_by, max_results=max_results)


@mcp_app.tool()
def get_offer_details(offer_id: int) -> str:
    """Get detailed information about a specific offer."""
    return _get_offer_details(_get_client(), offer_id=offer_id)


@mcp_app.tool()
def get_config() -> str:
    """Show the current vast-mcp configuration."""
    return _get_config(_get_state())


@mcp_app.tool()
def set_config(key: str, value: str) -> str:
    """Update a configuration field.

    Args:
        key: Config key (api_key, idle_threshold_hours, monitor_interval_minutes, default_instance_type, default_sort, default_max_results).
        value: New value (numeric values are converted automatically).
    """
    # Coerce to the correct type based on the existing field
    from vast_mcp.models import Config
    if key not in Config.__dataclass_fields__:
        return f"Error: Unknown config key: {key}"
    field_type = Config.__dataclass_fields__[key].type
    converted: int | float | str = value
    if field_type == "int":
        try:
            converted = int(value)
        except ValueError:
            return f"Error: {key} requires an integer value."
    elif field_type == "float":
        try:
            converted = float(value)
        except ValueError:
            return f"Error: {key} requires a numeric value."
    return _set_config(_get_state(), key=key, value=converted)


@mcp_app.tool()
def set_ssh_key(path: str) -> str:
    """Set the path to the SSH private key used for connecting to Vast.ai instances.

    Args:
        path: Absolute path to the SSH private key file (e.g., '~/.ssh/vastai_key').
    """
    from pathlib import Path as P
    expanded = P(path).expanduser().resolve()
    if not expanded.exists():
        return f"Error: File not found: {expanded}"
    if not expanded.is_file():
        return f"Error: Not a file: {expanded}"
    return _set_config(_get_state(), key="ssh_key_path", value=str(expanded))


@mcp_app.tool()
def get_ssh_command(instance_id: int) -> str:
    """Get the SSH command to connect to a tracked instance.

    Args:
        instance_id: The instance ID to connect to.
    """
    sm = _get_state()
    config = sm.load_config()
    instances = sm.load_instances()
    if instance_id not in instances:
        return f"Instance {instance_id} not tracked."
    inst = instances[instance_id]
    if inst.status != "running":
        return f"Instance {instance_id} is {inst.status}, not running."

    try:
        client = _get_client()
        ssh_url = client.ssh_url(instance_id)
    except Exception as e:
        return f"Error fetching SSH info: {e}"

    # Parse ssh://user@host:port into a usable command
    import re
    match = re.match(r"ssh://(\w+)@([^:]+):(\d+)", ssh_url)
    if match:
        user, host, port = match.groups()
    else:
        return f"SSH URL: {ssh_url} (could not parse into command)"

    key_flag = f" -i {config.ssh_key_path}" if config.ssh_key_path else ""
    no_key_note = "" if config.ssh_key_path else "\n  # No SSH key configured — run set_ssh_key first"

    return (
        f"SSH into instance {instance_id} ({inst.gpu_name} x{inst.num_gpus}):\n\n"
        f"  ssh -p {port}{key_flag} {user}@{host}{no_key_note}\n\n"
        f"SCP files:\n"
        f"  scp -P {port}{key_flag} local_file {user}@{host}:/path/"
    )


# ---------------------------------------------------------------------------
# Task 6: Instance Lifecycle internal functions
# ---------------------------------------------------------------------------


def _create_instance(sm: StateManager, client, offer_id: int, image: str, disk_space: float, **kwargs) -> str:
    """Create an instance, track it in state, and return a summary."""
    result = client.create_instance(offer_id, image, disk_space, **kwargs)
    instance_id = result.get("new_contract")
    if instance_id is None:
        return f"Instance creation failed. API response: {result}"

    # Fetch instance details from API
    all_instances = client.show_instances()
    instance_data = next((i for i in all_instances if i.get("id") == instance_id), None)

    if instance_data:
        inst = Instance(
            instance_id=instance_id,
            machine_id=instance_data.get("machine_id", 0),
            gpu_name=instance_data.get("gpu_name", "unknown"),
            num_gpus=instance_data.get("num_gpus", 1),
            image=instance_data.get("image_uuid", image),
            status=instance_data.get("actual_status", "provisioning"),
        )
    else:
        inst = Instance(
            instance_id=instance_id,
            machine_id=0,
            gpu_name="unknown",
            num_gpus=1,
            image=image,
            status="provisioning",
        )

    sm.save_instance(inst)
    return (
        f"Instance {instance_id} created.\n"
        f"  GPU: {inst.gpu_name} x{inst.num_gpus}\n"
        f"  Image: {inst.image}\n"
        f"  Status: {inst.status}"
    )


def _stop_instance(sm: StateManager, client, instance_id: int) -> str:
    """Stop a tracked instance."""
    instances = sm.load_instances()
    if instance_id not in instances:
        return f"Instance {instance_id} not tracked."
    try:
        client.stop_instance(instance_id)
    except Exception as e:
        return f"Error stopping instance {instance_id}: {e}"
    sm.update_instance_status(instance_id, "stopped")
    return f"Stopped instance {instance_id}."


def _start_instance(sm: StateManager, client, instance_id: int) -> str:
    """Start a tracked stopped instance."""
    instances = sm.load_instances()
    if instance_id not in instances:
        return f"Instance {instance_id} not tracked."
    try:
        client.start_instance(instance_id)
    except Exception as e:
        return f"Error starting instance {instance_id}: {e}"
    sm.update_instance_status(instance_id, "running")
    return f"Started instance {instance_id}."


def _destroy_instance(sm: StateManager, client, instance_id: int) -> str:
    """Destroy a tracked instance and remove it from state."""
    instances = sm.load_instances()
    if instance_id not in instances:
        return f"Instance {instance_id} not tracked."
    try:
        client.destroy_instance(instance_id)
    except Exception as e:
        return f"Error destroying instance {instance_id}: {e}"
    sm.remove_instance(instance_id)
    return f"Destroyed instance {instance_id}."


def _list_instances(sm: StateManager, client) -> str:
    """List all tracked instances, syncing status from the API."""
    instances = sm.load_instances()
    if not instances:
        return "No instances are currently tracked."

    # Sync status from API
    live_data = client.show_instances()
    live_by_id = {i.get("id"): i for i in live_data}
    now = _now_iso()
    for inst in instances.values():
        if inst.instance_id in live_by_id:
            inst.status = live_by_id[inst.instance_id].get("actual_status", inst.status)
        inst.last_checked = now

    sm._write_instances(instances)

    lines = ["Tracked Instances:"]
    lines.append(f"{'ID':<8} {'GPU':<20} {'#GPU':<6} {'Status':<14} {'Image':<30} Services")
    lines.append("-" * 100)
    for inst in instances.values():
        svc_names = ", ".join(s.name for s in inst.services) if inst.services else "-"
        lines.append(
            f"{inst.instance_id:<8} "
            f"{inst.gpu_name:<20} "
            f"{inst.num_gpus:<6} "
            f"{inst.status:<14} "
            f"{inst.image:<30} "
            f"{svc_names}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Task 6: MCP tool wrappers — Instance Lifecycle
# ---------------------------------------------------------------------------


@mcp_app.tool()
def create_instance(offer_id: int, image: str, disk_space: float) -> str:
    """Create a new GPU instance on Vast.ai."""
    return _create_instance(_get_state(), _get_client(), offer_id=offer_id, image=image, disk_space=disk_space)


@mcp_app.tool()
def stop_instance(instance_id: int) -> str:
    """Stop a running tracked instance."""
    return _stop_instance(_get_state(), _get_client(), instance_id=instance_id)


@mcp_app.tool()
def start_instance(instance_id: int) -> str:
    """Start a stopped tracked instance."""
    return _start_instance(_get_state(), _get_client(), instance_id=instance_id)


@mcp_app.tool()
def destroy_instance(instance_id: int) -> str:
    """Permanently destroy a tracked instance."""
    return _destroy_instance(_get_state(), _get_client(), instance_id=instance_id)


@mcp_app.tool()
def list_instances() -> str:
    """List all tracked GPU instances with their current status."""
    return _list_instances(_get_state(), _get_client())


# ---------------------------------------------------------------------------
# Task 7: Service Registry internal functions
# ---------------------------------------------------------------------------


def _register_service(
    sm: StateManager,
    instance_id: int,
    name: str,
    port: int,
    summary: str,
    endpoint: str | None = None,
) -> str:
    """Register or replace a service on a tracked instance."""
    instances = sm.load_instances()
    if instance_id not in instances:
        return f"Instance {instance_id} not tracked."

    inst = instances[instance_id]
    # Replace existing service with same name
    inst.services = [s for s in inst.services if s.name != name]
    inst.services.append(Service(name=name, port=port, summary=summary, endpoint=endpoint))
    sm.save_instance(inst)
    return f"Registered service '{name}' on instance {instance_id} (port {port})."


def _unregister_service(sm: StateManager, instance_id: int, service_name: str) -> str:
    """Remove a service from a tracked instance."""
    instances = sm.load_instances()
    if instance_id not in instances:
        return f"Instance {instance_id} not tracked."

    inst = instances[instance_id]
    original_count = len(inst.services)
    inst.services = [s for s in inst.services if s.name != service_name]
    if len(inst.services) == original_count:
        return f"Service '{service_name}' not found on instance {instance_id}."

    sm.save_instance(inst)
    return f"Removed service '{service_name}' from instance {instance_id}."


def _probe_service(svc: Service) -> str:
    """Probe an HTTP endpoint and return a status string."""
    if not svc.endpoint:
        return "no endpoint"
    try:
        resp = httpx.get(svc.endpoint, timeout=5.0)
        if resp.status_code < 400:
            return f"up (HTTP {resp.status_code})"
        return f"down (HTTP {resp.status_code})"
    except Exception as e:
        return f"unreachable ({e})"


def _check_services(sm: StateManager, instance_id: int | None = None) -> str:
    """Probe service endpoints and report status. Updates last_alive on success."""
    instances = sm.load_instances()

    # Collect all (instance, service) pairs to check
    targets: list[tuple[Instance, Service]] = []
    if instance_id is not None:
        if instance_id not in instances:
            return f"Instance {instance_id} not tracked."
        for svc in instances[instance_id].services:
            targets.append((instances[instance_id], svc))
    else:
        for inst in instances.values():
            for svc in inst.services:
                targets.append((inst, svc))

    if not targets:
        return "No services registered."

    lines = ["Service Health Check:"]
    now = _now_iso()
    for inst, svc in targets:
        status = _probe_service(svc)
        if status.startswith("up"):
            svc.last_alive = now
        lines.append(f"  [{inst.instance_id}] {svc.name} (port {svc.port}): {status}")

    # Persist updated last_alive values
    sm._write_instances(instances)
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Task 7: MCP tool wrappers — Service Registry
# ---------------------------------------------------------------------------


@mcp_app.tool()
def register_service(instance_id: int, name: str, port: int, summary: str, endpoint: str | None = None) -> str:
    """Register a service on a tracked instance."""
    return _register_service(_get_state(), instance_id=instance_id, name=name, port=port, summary=summary, endpoint=endpoint)


@mcp_app.tool()
def unregister_service(instance_id: int, service_name: str) -> str:
    """Remove a service from a tracked instance."""
    return _unregister_service(_get_state(), instance_id=instance_id, service_name=service_name)


@mcp_app.tool()
def check_services(instance_id: int | None = None) -> str:
    """Check the health of registered services."""
    return _check_services(_get_state(), instance_id=instance_id)


# ---------------------------------------------------------------------------
# Task 8: Experiment Template internal functions
# ---------------------------------------------------------------------------


def _save_experiment(sm: StateManager, name: str, specs: dict, summary: str, image: str | None = None) -> str:
    """Save an experiment template."""
    experiment_specs = ExperimentSpecs.from_dict(specs)
    experiment = Experiment(name=name, summary=summary, specs=experiment_specs, image=image)
    sm.save_experiment(experiment)
    return f"Saved experiment '{name}'."


def _list_experiments(sm: StateManager) -> str:
    """List all saved experiment templates."""
    experiments = sm.load_experiments()
    if not experiments:
        return "No experiments saved."

    lines = ["Saved Experiments:"]
    for exp in experiments.values():
        lines.append(f"  {exp.name}: {exp.summary}")
        q = exp.specs.to_query_string()
        if q:
            lines.append(f"    Query: {q}")
        if exp.image:
            lines.append(f"    Image: {exp.image}")
    return "\n".join(lines)


def _load_experiment(sm: StateManager, client, name: str, overrides: dict | None = None) -> str:
    """Load an experiment template, apply overrides, and search for matching offers."""
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
        return f"Experiment '{name}' loaded. No matching offers found.\nQuery: {query}"

    lines = [f"Experiment '{name}' — Matching Offers:"]
    lines.append(f"Query: {query}")
    lines.append(f"{'ID':<8} {'GPU':<20} {'#GPU':<6} {'VRAM(MB)':<10} {'RAM(MB)':<10} {'Disk(GB)':<10} {'$/hr':<8} {'Reliability'}")
    lines.append("-" * 90)
    for o in offers:
        lines.append(
            f"{o.get('id', ''):<8} "
            f"{o.get('gpu_name', ''):<20} "
            f"{o.get('num_gpus', ''):<6} "
            f"{o.get('gpu_ram', ''):<10} "
            f"{o.get('cpu_ram', ''):<10} "
            f"{o.get('disk_space', ''):<10} "
            f"{o.get('dph_total', ''):<8} "
            f"{o.get('reliability', '')}"
        )
    return "\n".join(lines)


def _delete_experiment(sm: StateManager, name: str) -> str:
    """Delete a saved experiment template."""
    try:
        sm.delete_experiment(name)
        return f"Deleted experiment '{name}'."
    except KeyError:
        return f"Experiment '{name}' not found."


# ---------------------------------------------------------------------------
# Task 8: MCP tool wrappers — Experiment Templates
# ---------------------------------------------------------------------------


@mcp_app.tool()
def save_experiment(name: str, specs: str, summary: str, image: str | None = None) -> str:
    """Save an experiment template.

    Args:
        name: Template name (e.g., 'llama-finetune').
        specs: JSON string of specs (e.g., '{"gpu_name": "RTX_4090", "num_gpus": 2}').
        summary: Short description.
        image: Optional Docker image.
    """
    try:
        specs_dict = json_module.loads(specs)
    except json_module.JSONDecodeError as e:
        return f"Error: Invalid JSON in specs: {e}"
    return _save_experiment(_get_state(), name=name, specs=specs_dict, summary=summary, image=image)


@mcp_app.tool()
def list_experiments() -> str:
    """List all saved experiment templates."""
    return _list_experiments(_get_state())


@mcp_app.tool()
def load_experiment(name: str, overrides: str | None = None) -> str:
    """Load an experiment template and search for matching offers.

    Args:
        name: Template name to load.
        overrides: Optional JSON string of spec overrides (e.g., '{"num_gpus": 4}').
    """
    overrides_dict = None
    if overrides:
        try:
            overrides_dict = json_module.loads(overrides)
        except json_module.JSONDecodeError as e:
            return f"Error: Invalid JSON in overrides: {e}"
    return _load_experiment(_get_state(), _get_client(), name=name, overrides=overrides_dict)


@mcp_app.tool()
def delete_experiment(name: str) -> str:
    """Delete a saved experiment template."""
    return _delete_experiment(_get_state(), name=name)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    mcp_app.run(transport="stdio")
