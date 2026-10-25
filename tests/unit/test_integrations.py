"""Order provider and input validators."""
import pytest

from integrations import order_provider as op
from integrations.validators import (
    sanitize_address, sanitize_text, validate_email, validate_order_id,
)


# ── Validators ───────────────────────────────────────────────────────────────

@pytest.mark.parametrize("raw,expected", [
    ("ORD-12345", "ORD-12345"),
    ("ord-12345", "ORD-12345"),
    (" ORD12345 ", "ORD-12345"),
    ("12345", "ORD-12345"),
    ("ORD-1234", None),
    ("1234", None),
    ("my order is ORD-12345", None),
    ("", None),
    (None, None),
])
def test_validate_order_id(raw, expected):
    assert validate_order_id(raw) == expected


@pytest.mark.parametrize("raw,expected", [
    ("Alex@Example.com ", "alex@example.com"),
    ("no-at-sign.com", None),
    ("a@b", None),
    (None, None),
])
def test_validate_email(raw, expected):
    assert validate_email(raw) == expected


def test_sanitize_address():
    assert sanitize_address("  12 High St, Nicosia ") == "12 High St, Nicosia"
    assert sanitize_address("abc") is None                      # too short
    assert sanitize_address("12 <script> St") is None           # unsafe chars
    assert sanitize_address("x" * 201) is None                  # too long


def test_sanitize_text_strips_and_truncates():
    assert sanitize_text("broken {screen}") == "broken screen"
    assert len(sanitize_text("a" * 900)) == 500
    assert sanitize_text("<>") is None


# ── Mock provider ────────────────────────────────────────────────────────────

def test_mock_get_order_is_case_insensitive():
    p = op.MockOrderProvider()
    assert p.get_order("ord-12345")["status"] == "Shipped"
    assert p.get_order("ORD-00000") is None


def test_mock_find_by_email():
    orders = op.MockOrderProvider().find_orders_by_email(" Alex@Example.com ")
    assert [o["order_id"] for o in orders] == ["ORD-12345"]


def test_mock_update_address(monkeypatch):
    monkeypatch.setitem(op._MOCK_DB, "ORD-99999", {"address": "old"})
    p = op.MockOrderProvider()
    assert p.update_address("ORD-99999", "new") is True
    assert op._MOCK_DB["ORD-99999"]["address"] == "new"
    assert p.update_address("ORD-00000", "x") is False


# ── REST provider (HTTP mocked) ──────────────────────────────────────────────

class FakeResponse:
    def __init__(self, status=200, payload=None):
        self.status_code = status
        self._payload = payload

    def json(self):
        return self._payload

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(f"HTTP {self.status_code}")


class FakeSession:
    def __init__(self, responses):
        self.headers = {}
        self.responses = responses
        self.calls = []

    def get(self, url, params=None, timeout=None):
        self.calls.append(("GET", url, params))
        return self.responses.pop(0)

    def patch(self, url, json=None, timeout=None):
        self.calls.append(("PATCH", url, json))
        return self.responses.pop(0)


@pytest.fixture
def rest(monkeypatch):
    def make(*responses):
        session = FakeSession(list(responses))
        monkeypatch.setattr(op.http_lib, "Session", lambda: session)
        return op.RestOrderProvider("https://api.example.com/v1/", api_key="k"), session
    return make


def test_rest_sends_bearer_token_and_strips_slash(rest):
    p, session = rest(FakeResponse(200, {"status": "Shipped"}))
    assert p.get_order("ORD-12345") == {"status": "Shipped"}
    assert session.headers["Authorization"] == "Bearer k"
    assert session.calls == [("GET", "https://api.example.com/v1/orders/ORD-12345", None)]


def test_rest_404_and_errors_return_none(rest):
    p, _ = rest(FakeResponse(404), FakeResponse(500))
    assert p.get_order("ORD-1") is None
    assert p.get_order("ORD-2") is None


@pytest.mark.parametrize("payload", [
    [{"id": "ORD-1", "created_at": "2026-01-01"}, {"id": "ORD-2", "created_at": "2026-02-01"}],
    {"orders": [{"id": "ORD-1", "created_at": "2026-01-01"}, {"id": "ORD-2", "created_at": "2026-02-01"}]},
    {"data": [{"order_id": "ORD-1", "created_at": "2026-01-01"}, {"order_id": "ORD-2", "created_at": "2026-02-01"}]},
])
def test_rest_find_by_email_normalises_shapes(rest, payload):
    p, _ = rest(FakeResponse(200, payload))
    assert [o["order_id"] for o in p.find_orders_by_email("a@b.co")] == ["ORD-2", "ORD-1"]


def test_rest_update_address(rest):
    p, session = rest(FakeResponse(200), FakeResponse(422))
    assert p.update_address("ORD-1", "12 High St") is True
    assert session.calls[0] == ("PATCH", "https://api.example.com/v1/orders/ORD-1", {"address": "12 High St"})
    assert p.update_address("ORD-1", "x") is False


# ── Factory ──────────────────────────────────────────────────────────────────

def test_factory_requires_base_url_for_rest(monkeypatch):
    monkeypatch.setattr(op, "_provider", None)
    monkeypatch.setenv("ORDER_PROVIDER", "rest")
    monkeypatch.delenv("ORDER_API_BASE_URL", raising=False)
    with pytest.raises(EnvironmentError):
        op.get_order_provider()


def test_factory_defaults_to_mock(monkeypatch):
    monkeypatch.setattr(op, "_provider", None)
    monkeypatch.setenv("ORDER_PROVIDER", "mock")
    assert isinstance(op.get_order_provider(), op.MockOrderProvider)
