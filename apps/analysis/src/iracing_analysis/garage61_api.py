"""Thin client for the official Garage 61 REST API (v1).

Authenticated with a personal access token (garage61.net → My applications).
Polite by design: a minimum interval between requests plus exponential
backoff honouring ``Retry-After`` — the developer docs ask applications not
to flood the server, and a harvester has no reason to hurry.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable

import httpx

__all__ = ["Garage61Client", "Garage61Error"]

log = logging.getLogger(__name__)

_RETRYABLE = {429, 500, 502, 503, 504}
_MAX_RETRIES = 4


class Garage61Error(RuntimeError):
    def __init__(self, status: int, message: str):
        super().__init__(f"Garage 61 API error {status}: {message}")
        self.status = status


class Garage61Client:
    def __init__(
        self,
        token: str,
        base_url: str = "https://garage61.net/api/v1",
        min_interval_s: float = 2.0,
        timeout_s: float = 60.0,
        _sleep: Callable[[float], None] = time.sleep,
        _clock: Callable[[], float] = time.monotonic,
    ):
        self._base_url = base_url.rstrip("/")
        self._min_interval = min_interval_s
        self._sleep = _sleep
        self._clock = _clock
        self._last_request = -1e9
        self._http = httpx.Client(
            headers={"Authorization": f"Bearer {token}"}, timeout=timeout_s
        )

    # -- plumbing ---------------------------------------------------------

    def _throttle(self) -> None:
        wait = self._min_interval - (self._clock() - self._last_request)
        if wait > 0:
            self._sleep(wait)
        self._last_request = self._clock()

    def _get(self, path: str, params: dict[str, Any] | None = None) -> httpx.Response:
        encoded = {}
        for key, value in (params or {}).items():
            if value is None:
                continue
            if isinstance(value, bool):
                encoded[key] = "true" if value else "false"
            elif isinstance(value, (list, tuple)):
                encoded[key] = ",".join(str(v) for v in value)
            else:
                encoded[key] = str(value)

        for attempt in range(_MAX_RETRIES + 1):
            self._throttle()
            response = self._http.get(f"{self._base_url}{path}", params=encoded)
            if response.status_code < 400:
                return response
            if response.status_code in _RETRYABLE and attempt < _MAX_RETRIES:
                retry_after = response.headers.get("Retry-After")
                delay = float(retry_after) if retry_after else max(self._min_interval, 1.0) * 2**attempt
                log.warning("HTTP %s on %s — retrying in %.1fs", response.status_code, path, delay)
                self._sleep(delay)
                continue
            if response.status_code == 401:
                raise Garage61Error(401, "invalid or expired token (create one under My applications)")
            if response.status_code in (402, 403):
                raise Garage61Error(
                    response.status_code,
                    "access denied — telemetry visibility filters (seeTelemetry) and "
                    "other drivers' laps require a Pro plan and lap visibility "
                    "(teammates / followed drivers)",
                )
            raise Garage61Error(response.status_code, response.text[:300])
        raise Garage61Error(599, "retries exhausted")  # pragma: no cover

    def _get_items(self, path: str) -> list[dict]:
        return self._get(path).json().get("items", [])

    # -- endpoints -----------------------------------------------------------

    def cars(self) -> list[dict]:
        return self._get_items("/cars")

    def tracks(self) -> list[dict]:
        return self._get_items("/tracks")

    def car_groups(self) -> list[dict]:
        return self._get_items("/car-groups")

    def find_laps(self, page_size: int = 1000, **filters: Any) -> list[dict]:
        """All laps matching ``filters`` (see the developer docs for the
        parameter list), transparently paginated."""
        items: list[dict] = []
        offset = 0
        while True:
            payload = self._get(
                "/laps", {**filters, "limit": page_size, "offset": offset}
            ).json()
            page = payload.get("items", [])
            items.extend(page)
            total = payload.get("total", len(items))
            offset += len(page)
            if not page or offset >= total:
                return items

    def download_lap_csv(self, lap_id: str) -> bytes:
        return self._get(f"/laps/{lap_id}/csv").content
