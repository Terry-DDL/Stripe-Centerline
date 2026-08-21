"""Low-overhead wall-clock profiling for one desktop analysis request."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from functools import wraps
import time


@dataclass
class ProfileSession:
    """Collect nested perf-counter intervals without changing call results."""

    events: list[dict] = field(default_factory=list)
    stack: list[dict] = field(default_factory=list)

    def report(self) -> dict:
        categories = {}
        calls = {}
        for event in self.events:
            label = event["label"]
            calls[label] = calls.get(label, 0) + 1
            category = event.get("category")
            if category is not None:
                categories[category] = (
                    categories.get(category, 0.0) + event["self_ms"]
                )
        return {
            "timer": "time.perf_counter",
            "categories_ms": {
                key: round(value, 3)
                for key, value in sorted(categories.items())
            },
            "call_counts": dict(sorted(calls.items())),
            "events": list(self.events),
        }


_ACTIVE_SESSION: ContextVar[ProfileSession | None] = ContextVar(
    "stripe_active_profile_session",
    default=None,
)


def active_session() -> ProfileSession | None:
    """Return the request-local profiling session, when profiling is active."""

    return _ACTIVE_SESSION.get()


@contextmanager
def activate(session: ProfileSession):
    """Make one session active for the current worker context."""

    token = _ACTIVE_SESSION.set(session)
    try:
        yield session
    finally:
        _ACTIVE_SESSION.reset(token)


@contextmanager
def stage(label: str, category: str | None = None):
    """Record inclusive and self wall time for one existing function boundary."""

    session = active_session()
    if session is None:
        yield
        return
    frame = {
        "label": label,
        "category": category,
        "child_seconds": 0.0,
        "depth": len(session.stack),
    }
    session.stack.append(frame)
    started = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - started
        session.stack.pop()
        self_seconds = max(0.0, elapsed - frame["child_seconds"])
        session.events.append(
            {
                "label": label,
                "category": category,
                "depth": frame["depth"],
                "elapsed_ms": round(elapsed * 1000.0, 3),
                "self_ms": round(self_seconds * 1000.0, 3),
            }
        )
        if session.stack:
            session.stack[-1]["child_seconds"] += elapsed


def profiled(label: str, category: str | None = None):
    """Decorate an existing function boundary with the request-local timer."""

    def decorator(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            with stage(label, category):
                return function(*args, **kwargs)

        return wrapped

    return decorator
