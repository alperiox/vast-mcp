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
