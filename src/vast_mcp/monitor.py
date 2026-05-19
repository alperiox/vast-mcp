from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

import httpx

from vast_mcp.models import _now_iso
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
                        if resp.status_code < 400:
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
                if svc.last_alive is not None:
                    last = datetime.fromisoformat(svc.last_alive)
                else:
                    # Never been alive — use registration time as baseline
                    last = datetime.fromisoformat(svc.registered_at)
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
