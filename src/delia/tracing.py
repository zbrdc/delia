# Copyright (C) 2024 Delia Contributors
# SPDX-License-Identifier: GPL-3.0-or-later

"""
Lightweight Tracing for Delia Orchestration.

Provides simple, OpenTelemetry-inspired tracing without the heavyweight
dependencies. Traces are output via structlog for easy debugging.

Usage:
    from delia.tracing import trace, get_current_trace
    
    async def process_request():
        with trace("orchestration", task_type="review") as t:
            t.event("intent_detected", mode="voting")
            
            with trace("voting") as voting_trace:
                voting_trace.event("vote_added", model="qwen")
                ...
            
            t.event("complete", tokens=1500)
        
        # Trace automatically logged on exit

Key Features:
- Nested spans with parent-child relationships
- Automatic timing (start/end/duration)
- Event recording within spans
- Automatic structlog output
- Thread-safe via contextvars
- Zero external dependencies
"""

from __future__ import annotations

import contextvars
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generator

import structlog

log = structlog.get_logger()


# =============================================================================
# Trace/Span Data Structures
# =============================================================================

@dataclass
class SpanEvent:
    """An event within a span."""
    
    name: str
    timestamp: float = field(default_factory=time.time)
    attributes: dict[str, Any] = field(default_factory=dict)


@dataclass
class Span:
    """A single span in a trace."""
    
    name: str
    trace_id: str
    span_id: str = field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_id: str | None = None
    start_time: float = field(default_factory=time.time)
    end_time: float | None = None
    attributes: dict[str, Any] = field(default_factory=dict)
    events: list[SpanEvent] = field(default_factory=list)
    status: str = "ok"  # ok, error
    error_message: str | None = None
    
    @property
    def duration_ms(self) -> int:
        """Get span duration in milliseconds."""
        if self.end_time is None:
            return int((time.time() - self.start_time) * 1000)
        return int((self.end_time - self.start_time) * 1000)
    
    def event(self, name: str, **attributes: Any) -> None:
        """Record an event within this span."""
        self.events.append(SpanEvent(name=name, attributes=attributes))
    
    def set_error(self, message: str) -> None:
        """Mark span as errored."""
        self.status = "error"
        self.error_message = message
    
    def set_attribute(self, key: str, value: Any) -> None:
        """Set a span attribute."""
        self.attributes[key] = value
    
    def finish(self) -> None:
        """Mark span as finished."""
        self.end_time = time.time()
    
    def to_dict(self) -> dict[str, Any]:
        """Convert span to dictionary for logging."""
        return {
            "trace_id": self.trace_id,
            "span_id": self.span_id,
            "parent_id": self.parent_id,
            "name": self.name,
            "duration_ms": self.duration_ms,
            "status": self.status,
            "attributes": self.attributes,
            "events": [
                {"name": e.name, **e.attributes}
                for e in self.events
            ],
            "error": self.error_message,
        }


@dataclass
class Trace:
    """A complete trace containing multiple spans."""
    
    trace_id: str = field(default_factory=lambda: uuid.uuid4().hex[:32])
    spans: list[Span] = field(default_factory=list)
    root_span: Span | None = None
    
    @property
    def duration_ms(self) -> int:
        """Get total trace duration."""
        if self.root_span:
            return self.root_span.duration_ms
        return 0
    
    def to_dict(self) -> dict[str, Any]:
        """Convert trace to dictionary for logging."""
        return {
            "trace_id": self.trace_id,
            "duration_ms": self.duration_ms,
            "span_count": len(self.spans),
            "spans": [s.to_dict() for s in self.spans],
        }


# =============================================================================
# Context Management
# =============================================================================

# Current trace context (thread-safe)
_current_trace: contextvars.ContextVar[Trace | None] = contextvars.ContextVar(
    "current_trace", default=None
)
_current_span: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
    "current_span", default=None
)


def get_current_trace() -> Trace | None:
    """Get the current trace from context."""
    return _current_trace.get()


def get_current_span() -> Span | None:
    """Get the current span from context."""
    return _current_span.get()


@contextmanager
def trace(name: str, **attributes: Any) -> Generator[Span, None, None]:
    """
    Context manager for creating a traced span.
    
    Automatically handles:
    - Creating new trace if needed
    - Nesting spans with parent-child relationships
    - Timing (start/end/duration)
    - Logging trace on completion
    
    Args:
        name: Name of the span (e.g., "orchestration", "voting", "llm_call")
        **attributes: Initial attributes to attach to the span
        
    Yields:
        Span object for adding events and attributes
        
    Example:
        with trace("orchestration", task="review") as span:
            span.event("intent_detected", mode="voting")
            # ... do work ...
            span.event("complete", tokens=1500)
    """
    current_trace = _current_trace.get()
    parent_span = _current_span.get()
    
    # Create new trace if we're not inside one
    is_root = current_trace is None
    if is_root:
        current_trace = Trace()
        _current_trace.set(current_trace)
    
    # Create span
    span = Span(
        name=name,
        trace_id=current_trace.trace_id,
        parent_id=parent_span.span_id if parent_span else None,
        attributes=attributes,
    )
    
    # Set as root span if first
    if is_root:
        current_trace.root_span = span
    
    current_trace.spans.append(span)
    
    # Set as current span
    token = _current_span.set(span)
    
    try:
        yield span
    except Exception as e:
        span.set_error(str(e))
        raise
    finally:
        span.finish()
        _current_span.reset(token)
        
        # If this was the root span, log the trace
        if is_root:
            _log_trace(current_trace)
            _current_trace.set(None)


def _log_trace(trace: Trace) -> None:
    """Log a completed trace via structlog."""
    # Build a summary for the main log line
    root = trace.root_span
    if not root:
        return
    
    # Collect key metrics
    span_names = [s.name for s in trace.spans]
    event_count = sum(len(s.events) for s in trace.spans)
    error_spans = [s.name for s in trace.spans if s.status == "error"]
    
    # Log based on status
    if error_spans:
        log.warning(
            "trace_complete",
            trace_id=trace.trace_id[:8],
            root=root.name,
            duration_ms=trace.duration_ms,
            spans=len(trace.spans),
            events=event_count,
            status="error",
            error_spans=error_spans,
            attributes=root.attributes,
        )
    else:
        log.debug(
            "trace_complete",
            trace_id=trace.trace_id[:8],
            root=root.name,
            duration_ms=trace.duration_ms,
            spans=len(trace.spans),
            events=event_count,
            status="ok",
            span_names=span_names,
        )


# =============================================================================
# Utility Functions
# =============================================================================

def add_event(name: str, **attributes: Any) -> None:
    """
    Add an event to the current span (if any).
    
    This is a convenience function for adding events without
    needing the span object directly.
    
    Args:
        name: Event name
        **attributes: Event attributes
    """
    span = _current_span.get()
    if span:
        span.event(name, **attributes)


def set_attribute(key: str, value: Any) -> None:
    """
    Set an attribute on the current span (if any).
    
    Args:
        key: Attribute name
        value: Attribute value
    """
    span = _current_span.get()
    if span:
        span.set_attribute(key, value)


def set_error(message: str) -> None:
    """
    Mark the current span as errored.
    
    Args:
        message: Error message
    """
    span = _current_span.get()
    if span:
        span.set_error(message)


# =============================================================================
# Decorators
# =============================================================================

def traced(name: str | None = None, **default_attributes: Any):
    """
    Decorator for tracing a function.
    
    Args:
        name: Span name (defaults to function name)
        **default_attributes: Attributes to attach to every call
        
    Example:
        @traced("llm_call", provider="ollama")
        async def call_model(prompt: str):
            ...
    """
    def decorator(func):
        span_name = name or func.__name__
        
        async def async_wrapper(*args, **kwargs):
            with trace(span_name, **default_attributes) as span:
                return await func(*args, **kwargs)
        
        def sync_wrapper(*args, **kwargs):
            with trace(span_name, **default_attributes) as span:
                return func(*args, **kwargs)
        
        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return sync_wrapper
    
    return decorator

