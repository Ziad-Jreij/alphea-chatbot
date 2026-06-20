"""Order data integration layer.

To plug in a real order system, set ORDER_PROVIDER=rest in your .env and
configure ORDER_API_BASE_URL + ORDER_API_KEY. The REST provider expects:

  GET  {base_url}/orders/{order_id}          → order object
  GET  {base_url}/orders?email={email}        → list OR {"orders": [...]}
  PATCH {base_url}/orders/{order_id}          → body {"address": "..."}

Each order object must contain: status, eta, created_at, address, email.
Add an 'order_id' key (or 'id') so find_orders_by_email can identify records.

For any other backend (Shopify, WooCommerce, SAP) subclass OrderProvider and
register the type string in get_order_provider().
"""
import os
import logging
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from typing import Optional

import requests as http_lib

log = logging.getLogger(__name__)

# ── Mock data (demo / development) ───────────────────────────────────────────
_NOW = datetime.now()

_MOCK_DB: dict[str, dict] = {
    "ORD-12345": {
        "status": "Shipped",
        "eta": "2 days",
        "created_at": (_NOW - timedelta(days=2)).isoformat(),
        "address": "123 Main St, Nicosia, 10001",
        "email": "alex@example.com",
        "total": 89.90,
    },
    "ORD-98765": {
        "status": "Processing",
        "eta": "4 days",
        "created_at": (_NOW - timedelta(hours=3)).isoformat(),
        "address": "45 Green Rd, Larnaca, 6023",
        "email": "maria@example.com",
        "total": 149.50,
    },
    "ORD-55555": {
        "status": "Delivered",
        "eta": "Delivered",
        "created_at": (_NOW - timedelta(days=5)).isoformat(),
        "address": "10 Beach Ave, Limassol, 3300",
        "email": "john@example.com",
        "total": 59.00,
    },
    "ORD-77777": {
        "status": "On hold",
        "eta": "TBD",
        "created_at": (_NOW - timedelta(hours=10)).isoformat(),
        "address": "78 Lake St, Paphos, 8100",
        "email": "sofia@example.com",
        "total": 220.00,
    },
}


# ── Abstract interface ────────────────────────────────────────────────────────
class OrderProvider(ABC):
    """Interface for order data access. One implementation per backend type."""

    @abstractmethod
    def get_order(self, order_id: str) -> Optional[dict]:
        """Return order dict or None if not found."""

    @abstractmethod
    def find_orders_by_email(self, email: str) -> list:
        """Return orders for this email, each with an 'order_id' key, newest first."""

    @abstractmethod
    def update_address(self, order_id: str, new_address: str) -> bool:
        """Update the delivery address. Return True on success."""


# ── Mock provider ─────────────────────────────────────────────────────────────
class MockOrderProvider(OrderProvider):
    """In-memory provider for development and demo. State resets on restart."""

    def get_order(self, order_id: str) -> Optional[dict]:
        return _MOCK_DB.get(order_id.upper())

    def find_orders_by_email(self, email: str) -> list:
        email = email.strip().lower()
        matches = [
            {**data, "order_id": oid}
            for oid, data in _MOCK_DB.items()
            if (data.get("email") or "").strip().lower() == email
        ]
        matches.sort(key=lambda x: x.get("created_at", ""), reverse=True)
        return matches

    def update_address(self, order_id: str, new_address: str) -> bool:
        rec = _MOCK_DB.get(order_id.upper())
        if rec is None:
            return False
        rec["address"] = new_address
        return True


# ── REST provider ─────────────────────────────────────────────────────────────
class RestOrderProvider(OrderProvider):
    """Generic REST adapter. Set ORDER_API_BASE_URL and ORDER_API_KEY."""

    def __init__(self, base_url: str, api_key: str = "", timeout: int = 10):
        self.base_url = base_url.rstrip("/")
        self.timeout  = timeout
        self._session = http_lib.Session()
        self._session.headers.update({"Content-Type": "application/json"})
        if api_key:
            self._session.headers["Authorization"] = f"Bearer {api_key}"

    def _get(self, path: str, params: dict = None) -> Optional[object]:
        url = f"{self.base_url}{path}"
        try:
            resp = self._session.get(url, params=params, timeout=self.timeout)
            if resp.status_code == 404:
                return None
            resp.raise_for_status()
            return resp.json()
        except Exception as exc:
            log.error("RestOrderProvider GET %s failed: %s", url, exc)
            return None

    def _patch(self, path: str, body: dict) -> bool:
        url = f"{self.base_url}{path}"
        try:
            resp = self._session.patch(url, json=body, timeout=self.timeout)
            resp.raise_for_status()
            return True
        except Exception as exc:
            log.error("RestOrderProvider PATCH %s failed: %s", url, exc)
            return False

    def get_order(self, order_id: str) -> Optional[dict]:
        return self._get(f"/orders/{order_id}")

    def find_orders_by_email(self, email: str) -> list:
        result = self._get("/orders", params={"email": email})
        orders = []
        if isinstance(result, list):
            orders = result
        elif isinstance(result, dict):
            orders = result.get("orders") or result.get("data") or []
        # Normalise: ensure each item has order_id
        for o in orders:
            if "order_id" not in o and "id" in o:
                o["order_id"] = o["id"]
        return sorted(orders, key=lambda x: x.get("created_at", ""), reverse=True)

    def update_address(self, order_id: str, new_address: str) -> bool:
        return self._patch(f"/orders/{order_id}", {"address": new_address})


# ── Factory (singleton) ───────────────────────────────────────────────────────
_provider: Optional[OrderProvider] = None


def get_order_provider() -> OrderProvider:
    """Return the configured order provider (lazy singleton)."""
    global _provider
    if _provider is None:
        ptype = os.environ.get("ORDER_PROVIDER", "mock").lower()
        if ptype == "rest":
            base_url = os.environ.get("ORDER_API_BASE_URL", "")
            if not base_url:
                raise EnvironmentError(
                    "ORDER_PROVIDER=rest requires ORDER_API_BASE_URL to be set."
                )
            _provider = RestOrderProvider(
                base_url=base_url,
                api_key=os.environ.get("ORDER_API_KEY", ""),
                timeout=int(os.environ.get("ORDER_API_TIMEOUT", "10")),
            )
        else:
            _provider = MockOrderProvider()
        log.info("OrderProvider ready: %s", type(_provider).__name__)
    return _provider
