"""Input validation and sanitization for all user-supplied values."""
import re
from typing import Optional

ORDER_RE   = re.compile(r"^ORD-\d{5}$")
EMAIL_RE   = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")
_UNSAFE_RE = re.compile(r"[<>{}\[\]\\|^~`]")  # chars blocked in free-text fields

MAX_ADDRESS_LEN = 200
MAX_TEXT_LEN    = 500


def validate_order_id(s: Optional[str]) -> Optional[str]:
    """Normalize and validate an order ID.
    Accepts 'ORD-12345' or bare '12345' or 'ORD12345'. Returns uppercased or None."""
    if not s:
        return None
    s = s.strip().upper().replace(" ", "")
    if s.startswith("ORD") and "-" not in s and len(s) == 8 and s[3:].isdigit():
        s = f"ORD-{s[3:]}"
    # Also accept a bare 5-digit number
    if re.fullmatch(r"\d{5}", s):
        s = f"ORD-{s}"
    return s if ORDER_RE.fullmatch(s) else None


def validate_email(s: Optional[str]) -> Optional[str]:
    """Normalize and validate an email address. Returns lowercase or None."""
    if not s:
        return None
    s = s.strip().lower()
    return s if EMAIL_RE.fullmatch(s) else None


def sanitize_address(s: Optional[str]) -> Optional[str]:
    """Strip dangerous characters from a shipping address and enforce length.
    Returns cleaned string or None if invalid."""
    if not s:
        return None
    s = s.strip()
    if len(s) > MAX_ADDRESS_LEN or len(s) < 5:
        return None
    if _UNSAFE_RE.search(s):
        return None
    return s


def sanitize_text(s: Optional[str], max_len: int = MAX_TEXT_LEN) -> Optional[str]:
    """General-purpose sanitizer for free-text fields (issue descriptions, etc.).
    Truncates to max_len and strips unsafe chars."""
    if not s:
        return None
    s = s.strip()
    s = _UNSAFE_RE.sub("", s).strip()
    return s[:max_len] if s else None
