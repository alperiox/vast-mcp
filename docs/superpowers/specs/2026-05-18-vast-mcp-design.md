# Vast.ai MCP Server — Design Spec

## Overview

A Python MCP server for managing Vast.ai GPU instances through Claude Code (or any MCP client). The server provides tools for discovering machines, provisioning instances, tracking running services, saving experiment templates, and monitoring idle instances.

**Architecture:** One Python package, two entry points:
- `vast-mcp` — stdio MCP server (invoked by MCP clients)
- `vast-monitor` — cron script for periodic liveness checks and idle notifications

**Key principle:** The MCP always presents options and lets the user choose. It never auto-provisions or auto-destroys instances. The monitor only sends notifications — it never takes action.

## Technology Stack

- **Language:** Python 3.12+
- **MCP SDK:** `mcp` (FastMCP, stdio transport)
- **Vast.ai SDK:** `vastai` (pip package, wraps REST API)
- **Package manager:** `uv`
- **Build system:** `hatchling`
- **Persistence:** JSON files in `~/.vast-mcp/`
- **Notifications:** macOS native via `osascript`

## Project Structure

```
vast-mcp/
├── src/vast_mcp/
│   ├── __init__.py
│   ├── mcp_server.py          # FastMCP stdio server (all tools)
│   ├── monitor.py             # Cron entry point (liveness, idle detection, notifications)
│   ├── vast_client.py         # Thin wrapper around vastai SDK
│   ├── state.py               # JSON state read/write (instances, experiments, config)
│   ├── models.py              # Dataclasses: Instance, Service, Experiment, Config
│   └── notifications.py       # macOS notification helpers
├── skills/
│   ├── provision-instance.md  # Skill: guided provisioning workflow
│   └── save-experiment.md     # Skill: extract & save experiment from conversation
├── tests/
├── pyproject.toml
└── README.md
```

**Entry points (in `pyproject.toml`):**
- `vast-mcp = "vast_mcp.mcp_server:main"`
- `vast-monitor = "vast_mcp.monitor:main"`

## State Directory

All persistent state lives in `~/.vast-mcp/`:

```
~/.vast-mcp/
├── config.json        # API key, default filters, thresholds
├── instances.json     # Instance registry with service annotations
├── experiments.json   # Saved experiment templates
└── monitor.log        # Monitor run history
```

## Data Models

### Config (`config.json`)

```json
{
  "api_key": null,
  "idle_threshold_hours": 10,
  "monitor_interval_minutes": 30,
  "default_instance_type": "container",
  "default_sort": "dph_total",
  "default_max_results": 10
}
```

- `api_key`: Vast.ai API key. Falls back to `VAST_API_KEY` env var, then `~/.config/vastai/vast_api_key`.
- `idle_threshold_hours`: Hours with all services dead before notification. Default 10.
- `default_instance_type`: `"container"` (default), `"vm"`, or `"all"`.

### Instance Registry (`instances.json`)

```json
{
  "instances": {
    "12345": {
      "instance_id": 12345,
      "machine_id": 6789,
      "gpu_name": "RTX 4090",
      "num_gpus": 2,
      "image": "pytorch/pytorch:latest",
      "created_at": "2026-05-18T10:00:00Z",
      "status": "running",
      "last_checked": "2026-05-18T14:00:00Z",
      "services": [
        {
          "name": "vllm-llama3",
          "port": 8000,
          "endpoint": "http://x.x.x.x:8000/v1/completions",
          "summary": "vLLM serving Llama-3-8B",
          "registered_at": "2026-05-18T10:30:00Z",
          "last_alive": "2026-05-18T13:55:00Z"
        }
      ],
      "experiment_name": null
    }
  }
}
```

- `status` is synced from the Vast.ai API on every `list_instances()` call and by the monitor.
- `services` are user-annotated via the `register_service` tool.
- `last_alive` is updated by the monitor when an endpoint responds.

### Experiment Templates (`experiments.json`)

```json
{
  "experiments": {
    "llama-finetune": {
      "name": "llama-finetune",
      "summary": "Fine-tuning Llama-3-8B with LoRA on custom dataset",
      "specs": {
        "gpu_name": "RTX_4090",
        "num_gpus": 2,
        "gpu_ram_min": 24000,
        "cpu_ram_min": 32000,
        "disk_space_min": 100,
        "max_dph": 1.5
      },
      "image": "pytorch/pytorch:2.3-cuda12.1",
      "created_at": "2026-05-18T10:00:00Z"
    }
  }
}
```

## MCP Tools (15 total)

### Machine Discovery

| Tool | Parameters | Description |
|------|-----------|-------------|
| `search_offers` | `query`, `sort_by?`, `max_results?` | Search available machines. Applies `default_instance_type` filter. Returns formatted table. |
| `get_offer_details` | `offer_id` | Detailed info on a specific offer. |

### Instance Lifecycle

| Tool | Parameters | Description |
|------|-----------|-------------|
| `create_instance` | `offer_id`, `image`, `disk_space`, `...` | Provision from a selected offer. Requires explicit offer_id. |
| `stop_instance` | `instance_id` | Stop instance (storage charges continue). |
| `start_instance` | `instance_id` | Restart a stopped instance. |
| `destroy_instance` | `instance_id` | Fully terminate and delete. |
| `list_instances` | — | List all tracked instances with service annotations and status (synced from API). |

### Service Registry

| Tool | Parameters | Description |
|------|-----------|-------------|
| `register_service` | `instance_id`, `name`, `port`, `endpoint?`, `summary` | Annotate what's running on an instance. |
| `unregister_service` | `instance_id`, `service_name` | Remove a service annotation. |
| `check_services` | `instance_id?` | Check liveness of registered endpoints. All instances if no id given. |

### Experiment Templates

| Tool | Parameters | Description |
|------|-----------|-------------|
| `save_experiment` | `name`, `specs`, `summary`, `image?` | Save a machine spec template. |
| `list_experiments` | — | List saved templates. |
| `load_experiment` | `name`, `overrides?` | Load template, apply overrides, search matching offers. |
| `delete_experiment` | `name` | Remove a template. |

### Configuration

| Tool | Parameters | Description |
|------|-----------|-------------|
| `get_config` | — | Show current configuration. |
| `set_config` | `key`, `value` | Update a config value. |

## Monitor (Cron Job)

### Behavior

Runs every `monitor_interval_minutes` (default 30). Each run:

1. **Sync instance status** — calls Vast.ai API, updates `instances.json` with current status. Catches instances stopped/destroyed outside the MCP.
2. **Check service liveness** — for each registered endpoint, attempts HTTP request (or TCP connect for non-HTTP ports). Updates `last_alive` if responsive.
3. **Evaluate idle threshold** — if ALL services on a running instance have `last_alive` older than `idle_threshold_hours`, send a macOS notification.
4. **Clean stale entries** — remove instances from registry that no longer exist on Vast.ai.
5. **Log results** — append to `monitor.log`.

### Notifications

macOS native notifications via `osascript`:
- **Idle instance:** "Vast.ai: Instance 12345 (2x RTX 4090) — all services down for 11h. Still in use?"
- **Instance gone:** "Vast.ai: Instance 12345 no longer exists on Vast.ai. Removed from tracking."

### Installation

- `vast-monitor install` — creates a launchd plist (or crontab entry) for periodic execution
- `vast-monitor uninstall` — removes the scheduled job
- `vast-monitor run` — single manual run (for testing)

## Skills

### `provision-instance`

Guides the agent through provisioning:
1. Ask the user what they need (or reference a saved experiment)
2. Extract specs from conversation context
3. Call `search_offers` and present options as a table
4. User picks an offer
5. Call `create_instance` with selected offer
6. Prompt user to register services once the instance is running

### `save-experiment`

Guides the agent to extract and save experiment configs:
1. Review recent conversation context
2. Extract machine requirements mentioned by the user
3. Generate a short summary and structured spec object
4. Call `save_experiment` with extracted data
5. Confirm saved template with the user

## Authentication

API key resolution order:
1. `config.json` `api_key` field
2. `VAST_API_KEY` environment variable
3. `~/.config/vastai/vast_api_key` (vast CLI default)

## Constraints

- Default to container-based instances (configurable via `default_instance_type`)
- Never auto-provision or auto-destroy — always present options, always confirm
- Monitor only notifies — never takes action on instances
- The MCP server is stateless between invocations (reads state from JSON on each call)
