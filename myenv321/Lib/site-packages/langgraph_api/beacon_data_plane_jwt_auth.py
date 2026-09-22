"""Data-plane JWT authentication for Beacon OTLP exports."""

import time
from datetime import UTC, datetime, timedelta
from typing import Any

import jwt
from requests.sessions import Session

DATA_PLANE_AUTH_HEADER = "X-Data-Plane-Auth"
BEACON_OTEL_COLLECTOR_PURPOSE = "beacon-otel-collector"

_BEACON_OTEL_JWT_TTL_SECONDS = 60 * 60
_BEACON_OTEL_JWT_REFRESH_BEFORE_SECONDS = 60


def sign_data_plane_jwt(data_plane_id: str, organization_id: str, secret: str) -> str:
    now = datetime.now(UTC)
    return jwt.encode(
        {
            "data_plane_id": data_plane_id,
            "organization_id": organization_id,
            "purpose": BEACON_OTEL_COLLECTOR_PURPOSE,
            "iat": now,
            "exp": now + timedelta(seconds=_BEACON_OTEL_JWT_TTL_SECONDS),
        },
        secret,
        algorithm="HS256",
    )


class _BeaconOtelJWTCache:
    def __init__(self, data_plane_id: str, organization_id: str, secret: str) -> None:
        self._data_plane_id = data_plane_id
        self._organization_id = organization_id
        self._secret = secret
        self._cached_token: tuple[str, float] | None = None

    def get(self) -> str:
        cached_token = self._cached_token
        if cached_token is not None:
            token, refresh_at = cached_token
            if time.monotonic() < refresh_at:
                return token

        token = sign_data_plane_jwt(
            self._data_plane_id, self._organization_id, self._secret
        )
        refresh_at = time.monotonic() + (
            _BEACON_OTEL_JWT_TTL_SECONDS - _BEACON_OTEL_JWT_REFRESH_BEFORE_SECONDS
        )
        self._cached_token = (token, refresh_at)
        return token


class _DataPlaneAuthSession(Session):
    def __init__(self, cache: _BeaconOtelJWTCache) -> None:
        super().__init__()
        self._cache = cache

    def request(self, method: str, url: str, **kwargs: Any):
        headers = dict(kwargs.pop("headers", None) or {})
        headers[DATA_PLANE_AUTH_HEADER] = self._cache.get()
        return super().request(method, url, headers=headers, **kwargs)


def beacon_otel_session(
    data_plane_id: str, organization_id: str, secret: str
) -> Session:
    cache = _BeaconOtelJWTCache(data_plane_id, organization_id, secret)
    return _DataPlaneAuthSession(cache)
