"""Reliable MIDI transport and synchronization helpers for the Mixer UI.

The listener thread owns only the hardware connection and a bounded event queue. Streamlit
Session State remains the per-browser-session source of truth. The module deliberately keeps
all helpers free of Streamlit imports so they can be regression-tested without a running app.
"""
from __future__ import annotations

from collections import deque
from copy import deepcopy
from dataclasses import dataclass
import threading
import time
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple


@dataclass(frozen=True)
class ChannelSpec:
    key: str
    label: str
    minimum: int
    maximum: int
    step: int = 1


CHANNELS: Tuple[ChannelSpec, ...] = (
    ChannelSpec("base", "1: БАЗА", 0, 100),
    ChannelSpec("req", "2: ПОТРЕБ", 0, 100),
    ChannelSpec("add", "3: ДОП", 0, 100),
    ChannelSpec("shift", "4: СДВИГ", -12, 24),
    ChannelSpec("trans_req", "7: ТР.ПОТРЕБ", 0, 100),
    ChannelSpec("trans_add", "8: ТР.ДОП", 0, 100),
)
CHANNEL_BY_KEY = {c.key: c for c in CHANNELS}

# Common two's-complement / binary-offset values emitted by relative encoders.
_RELATIVE_CORE = {
    1, 2, 3, 4, 5, 6, 7, 8,
    120, 121, 122, 123, 124, 125, 126, 127,
    59, 60, 61, 62, 63, 65, 66, 67, 68, 69,
}

_RELATIVE_MODES = {
    "relative_legacy", "relative_twos", "relative_binary", "relative_mackie"
}
_VALID_MODES = {"auto", "absolute", *_RELATIVE_MODES}


def clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def decode_relative(raw: int, dialect: str = "relative_legacy") -> int:
    """Decode a relative CC value using an explicit dialect.

    MIDI has several incompatible relative-encoder conventions. Treating them as one mode
    makes one direction reverse or jump. The legacy dialect preserves the original app
    behavior; new bindings should select the controller's actual dialect in the sidebar.
    """
    raw = int(clamp(int(raw), 0, 127))
    dialect = str(dialect or "relative_legacy").lower()
    if dialect in {"relative_twos", "twos", "2s"}:
        if raw == 64 or raw == 0:
            return 0
        return raw if 1 <= raw <= 63 else raw - 128
    if dialect in {"relative_binary", "binary", "offset"}:
        return raw - 64
    if dialect in {"relative_mackie", "mackie", "mcu", "signed_bit"}:
        if raw == 64 or raw == 0:
            return 0
        return raw if 1 <= raw <= 63 else -(raw - 64)
    # Backward-compatible mixed decoder used by previous releases.
    if 1 <= raw <= 8:
        return raw
    if 120 <= raw <= 127:
        return raw - 128
    if 65 <= raw <= 69:
        return raw - 64
    if 59 <= raw <= 63:
        return raw - 64
    return 0


def absolute_to_value(raw: int, spec: ChannelSpec) -> int:
    raw = int(clamp(int(raw), 0, 127))
    span = spec.maximum - spec.minimum
    return int(round(spec.minimum + (raw / 127.0) * span))


def create_transport_state(queue_size: int = 512) -> Dict[str, Any]:
    """Create the mutable process-wide object stored by ``st.cache_resource``."""
    qsize = max(32, int(queue_size))
    return {
        "lock": threading.RLock(),
        "bindings": [None] * len(CHANNELS),
        "mapping": [],  # legacy compatibility
        "cc_modes": {},
        "mode_overrides": {},
        "samples": {},
        "queue": deque(maxlen=qsize),
        "queue_size": qsize,
        "seq": 0,
        "dropped": 0,
        "port": None,
        "preferred_port": None,
        "available_ports": [],
        "input_port": None,
        "reconnect_requested": False,
        "events": [],
        "last_raw": {},
        "last_channel_event": {},
        "listener_started": False,
        "listener_thread": None,
        "listener_heartbeat": 0.0,
        # Explicit learn prevents startup noise or the wrong physical control from occupying
        # a logical slot. Existing installations can opt back into sequential auto-binding.
        "learn_slot": None,
        "auto_bind": False,
        "binding_generation": 0,
        # A cached resource is shared by all Streamlit sessions. Exactly one browser session
        # may consume the physical queue at a time; otherwise two tabs steal events from each
        # other and produce apparently random slider resets.
        "consumer_owner": None,
        "consumer_seen": 0.0,
    }


def ensure_transport_state(state: Dict[str, Any]) -> Dict[str, Any]:
    """Migrate older cached dictionaries without losing bindings."""
    if "lock" not in state:
        state["lock"] = threading.RLock()
    lock = state["lock"]
    with lock:
        old_mapping = list(state.get("mapping") or [])
        bindings = list(state.get("bindings") or [])
        if len(bindings) != len(CHANNELS):
            bindings = (bindings + [None] * len(CHANNELS))[:len(CHANNELS)]
        if old_mapping and not any(bindings):
            for i, sig in enumerate(old_mapping[:len(CHANNELS)]):
                bindings[i] = sig
        state["bindings"] = bindings
        state["mapping"] = [s for s in bindings if s]
        state.setdefault("cc_modes", {})
        state.setdefault("mode_overrides", {})
        # Cached resource dictionaries from previous releases used the ambiguous name
        # ``relative``. Preserve behavior but expose the dialect explicitly.
        state["cc_modes"] = {
            str(sig): ("relative_legacy" if str(mode) == "relative" else str(mode))
            for sig, mode in dict(state.get("cc_modes") or {}).items()
        }
        state["mode_overrides"] = {
            str(sig): ("relative_legacy" if str(mode) == "relative" else str(mode))
            for sig, mode in dict(state.get("mode_overrides") or {}).items()
            if ("relative_legacy" if str(mode) == "relative" else str(mode)) in _VALID_MODES - {"auto"}
        }
        state.setdefault("samples", {})
        if not isinstance(state.get("queue"), deque):
            state["queue"] = deque(maxlen=max(32, int(state.get("queue_size", 512))))
        state.setdefault("queue_size", state["queue"].maxlen or 512)
        state.setdefault("seq", 0)
        state.setdefault("dropped", 0)
        state.setdefault("port", None)
        state.setdefault("preferred_port", None)
        state.setdefault("available_ports", [])
        state.setdefault("input_port", None)
        state.setdefault("reconnect_requested", False)
        state.setdefault("events", [])
        state.setdefault("last_raw", {})
        state.setdefault("last_channel_event", {})
        state.setdefault("listener_started", False)
        state.setdefault("listener_thread", None)
        state.setdefault("listener_heartbeat", 0.0)
        state.setdefault("learn_slot", None)
        # Legacy versions auto-bound sequentially. Keep that behavior only when an old
        # in-progress mapping exists; clean/new states use explicit learn mode.
        state.setdefault("auto_bind", bool(old_mapping and len(old_mapping) < len(CHANNELS)))
        state.setdefault("binding_generation", 0)
        state.setdefault("consumer_owner", None)
        state.setdefault("consumer_seen", 0.0)
    return state


def append_log(state: Dict[str, Any], message: str, limit: int = 100) -> None:
    ensure_transport_state(state)
    with state["lock"]:
        events = list(state.get("events") or [])
        events.append(str(message))
        state["events"] = events[-max(10, int(limit)):]


def _binding_index(state: Dict[str, Any], signal: str) -> Optional[int]:
    try:
        return state["bindings"].index(signal)
    except ValueError:
        return None


def _clear_signal_runtime(state: Dict[str, Any], signal: Optional[str]) -> None:
    if not signal:
        return
    state["samples"].pop(signal, None)
    state["cc_modes"].pop(signal, None)
    state["last_raw"].pop(signal, None)


def bind_signal(state: Dict[str, Any], signal: str, channel_index: Optional[int] = None) -> Optional[int]:
    """Bind one physical signal to a fixed logical slot.

    Rebinding replaces the prior signal in the slot without shifting any later slots and
    flushes events normalized under the obsolete mapping.
    """
    ensure_transport_state(state)
    with state["lock"]:
        current = _binding_index(state, signal)
        if current is not None and channel_index is None:
            return current
        if channel_index is None:
            try:
                channel_index = state["bindings"].index(None)
            except ValueError:
                return None
        idx = int(channel_index)
        if not 0 <= idx < len(CHANNELS):
            return None

        # Remove the signal from any previous slot and remove the previous occupant of target.
        for i, existing in enumerate(list(state["bindings"])):
            if existing == signal and i != idx:
                state["bindings"][i] = None
        displaced = state["bindings"][idx]
        if displaced and displaced != signal:
            _clear_signal_runtime(state, displaced)
        state["bindings"][idx] = signal
        _clear_signal_runtime(state, signal)
        state["mapping"] = [s for s in state["bindings"] if s]
        affected = {CHANNELS[idx].key}
        state["queue"] = deque(
            (e for e in state["queue"] if e.get("signal") not in {signal, displaced}
             and e.get("channel") not in affected),
            maxlen=state["queue"].maxlen,
        )
        state["binding_generation"] = int(state.get("binding_generation", 0)) + 1
        return idx


def rename_bound_signal(state: Dict[str, Any], old_signal: str, new_signal: str) -> Optional[int]:
    """Migrate a legacy signal id (``cc_74``) to channel-aware form (``cc_0_74``).

    If both ids are present because a user already re-learned the control, the explicit new
    binding wins and the stale legacy slot is cleared instead of creating duplicate bindings.
    """
    ensure_transport_state(state)
    old_signal, new_signal = str(old_signal), str(new_signal)
    if not old_signal or not new_signal or old_signal == new_signal:
        with state["lock"]:
            return _binding_index(state, new_signal)
    with state["lock"]:
        old_idx = _binding_index(state, old_signal)
        new_idx = _binding_index(state, new_signal)
        if old_idx is None:
            return new_idx

        old_override = state["mode_overrides"].pop(old_signal, None)
        if new_idx is not None and new_idx != old_idx:
            # A channel-aware binding already exists. Keep it and only remove the stale slot.
            state["bindings"][old_idx] = None
            if old_override and new_signal not in state["mode_overrides"]:
                state["mode_overrides"][new_signal] = old_override
                state["cc_modes"][new_signal] = old_override
            result_idx = new_idx
        else:
            state["bindings"][old_idx] = new_signal
            if old_override:
                state["mode_overrides"][new_signal] = old_override
                state["cc_modes"][new_signal] = old_override
            result_idx = old_idx

        _clear_signal_runtime(state, old_signal)
        # Keep the explicitly copied mode after clearing transient runtime for the new id.
        state["samples"].pop(new_signal, None)
        state["last_raw"].pop(new_signal, None)
        if new_signal in state["mode_overrides"]:
            state["cc_modes"][new_signal] = state["mode_overrides"][new_signal]
        else:
            state["cc_modes"].pop(new_signal, None)
        state["mapping"] = [sig for sig in state["bindings"] if sig]
        state["queue"] = deque(
            (e for e in state["queue"] if e.get("signal") not in {old_signal, new_signal}),
            maxlen=state["queue"].maxlen,
        )
        state["binding_generation"] = int(state.get("binding_generation", 0)) + 1
        return result_idx


def begin_learn(state: Dict[str, Any], channel_index: int) -> bool:
    ensure_transport_state(state)
    idx = int(channel_index)
    if not 0 <= idx < len(CHANNELS):
        return False
    with state["lock"]:
        state["learn_slot"] = idx
    return True


def cancel_learn(state: Dict[str, Any]) -> None:
    ensure_transport_state(state)
    with state["lock"]:
        state["learn_slot"] = None


def set_auto_bind(state: Dict[str, Any], enabled: bool) -> None:
    ensure_transport_state(state)
    with state["lock"]:
        state["auto_bind"] = bool(enabled)
        if enabled:
            state["learn_slot"] = None


def unbind_channel(state: Dict[str, Any], channel_index: int) -> Optional[str]:
    ensure_transport_state(state)
    with state["lock"]:
        idx = int(channel_index)
        if not 0 <= idx < len(CHANNELS):
            return None
        signal = state["bindings"][idx]
        state["bindings"][idx] = None
        state["mapping"] = [s for s in state["bindings"] if s]
        _clear_signal_runtime(state, signal)
        if signal:
            state["mode_overrides"].pop(signal, None)
        ch_key = CHANNELS[idx].key
        state["queue"] = deque(
            (e for e in state["queue"] if e.get("channel") != ch_key and e.get("signal") != signal),
            maxlen=state["queue"].maxlen,
        )
        if state.get("learn_slot") == idx:
            state["learn_slot"] = None
        state["binding_generation"] = int(state.get("binding_generation", 0)) + 1
        return signal


def clear_bindings(state: Dict[str, Any]) -> None:
    ensure_transport_state(state)
    with state["lock"]:
        state["bindings"] = [None] * len(CHANNELS)
        state["mapping"] = []
        state["cc_modes"] = {}
        state["mode_overrides"] = {}
        state["samples"] = {}
        state["last_raw"] = {}
        state["last_channel_event"] = {}
        state["learn_slot"] = None
        state["queue"].clear()
        state["binding_generation"] = int(state.get("binding_generation", 0)) + 1


def set_mode_override(state: Dict[str, Any], signal: str, mode: str) -> None:
    ensure_transport_state(state)
    mode = str(mode or "auto").lower()
    if mode == "relative":
        mode = "relative_legacy"
    if mode not in _VALID_MODES:
        mode = "auto"
    with state["lock"]:
        if mode == "auto":
            state["mode_overrides"].pop(signal, None)
            state["cc_modes"].pop(signal, None)
            state["samples"].pop(signal, None)
        else:
            state["mode_overrides"][signal] = mode
            state["cc_modes"][signal] = mode
            state["samples"].pop(signal, None)
        state["queue"] = deque(
            (e for e in state["queue"] if e.get("signal") != signal),
            maxlen=state["queue"].maxlen,
        )


def _resolve_mode(state: Dict[str, Any], signal: str, raw: int, message_type: str) -> Optional[str]:
    if message_type == "pitchwheel":
        state["cc_modes"][signal] = "absolute"
        return "absolute"
    override = state["mode_overrides"].get(signal)
    if override == "absolute" or override in _RELATIVE_MODES:
        state["cc_modes"][signal] = override
        return override

    samples = list(state["samples"].get(signal, []))
    samples.append(int(raw))
    samples = samples[-8:]
    state["samples"][signal] = samples
    if len(samples) < 2:
        return None

    # Any value outside the relative code bands is decisive evidence of an absolute control.
    if any(v not in _RELATIVE_CORE for v in samples):
        state["cc_modes"][signal] = "absolute"
        return "absolute"

    # Smooth one-step movement near an edge is an absolute fader, not an encoder. This is
    # checked before relative detection because values 1..8 and 120..127 overlap both worlds.
    if len(samples) >= 3:
        tail = samples[-3:]
        diffs = [tail[i + 1] - tail[i] for i in range(2)]
        if len(set(tail)) == 3 and all(abs(d) == 1 for d in diffs) and diffs[0] == diffs[1]:
            state["cc_modes"][signal] = "absolute"
            return "absolute"

    values = set(samples)
    # Distinct direction pairs reveal the dialect. This is intentionally conservative:
    # repeated edge values can also come from an absolute fader parked at 0/127.
    detected = None
    if values.issubset(set(range(1, 16)) | set(range(65, 80))) and \
            any(1 <= v <= 15 for v in values) and any(65 <= v <= 79 for v in values):
        detected = "relative_mackie"
    elif values.issubset(set(range(1, 16)) | set(range(112, 128))) and \
            any(1 <= v <= 15 for v in values) and any(112 <= v <= 127 for v in values):
        detected = "relative_twos"
    elif values.issubset(set(range(56, 64)) | set(range(65, 73))) and \
            any(56 <= v <= 63 for v in values) and any(65 <= v <= 72 for v in values):
        detected = "relative_binary"
    if detected and len(samples) >= 3:
        state["cc_modes"][signal] = detected
        return detected

    # Non-edge acceleration codes can identify a relative stream, but not always its exact
    # convention. Keep legacy behavior only for auto mode and recommend explicit selection.
    decoded = [decode_relative(v, "relative_legacy") for v in samples]
    if len(samples) >= 4 and all(d != 0 and abs(d) <= 8 for d in decoded):
        if any(v not in {1, 127, 63, 65} for v in samples):
            state["cc_modes"][signal] = "relative_legacy"
            return "relative_legacy"

    if len(samples) >= 6:
        state["cc_modes"][signal] = "absolute"
        return "absolute"
    return None


def record_midi_message(
    state: Dict[str, Any],
    *,
    signal: str,
    raw: int,
    message_type: str = "control_change",
    timestamp: Optional[float] = None,
) -> Dict[str, Any]:
    """Normalize one hardware message and enqueue it without touching Streamlit state."""
    ensure_transport_state(state)
    timestamp = float(time.time() if timestamp is None else timestamp)
    raw = int(clamp(int(raw), 0, 127))
    with state["lock"]:
        state["listener_heartbeat"] = timestamp
        idx = _binding_index(state, signal)
        newly_bound = False
        if idx is None:
            learn_slot = state.get("learn_slot")
            if learn_slot is not None:
                idx = bind_signal(state, signal, int(learn_slot))
                state["learn_slot"] = None
            elif state.get("auto_bind", False):
                idx = bind_signal(state, signal)
            else:
                return {"status": "ignored", "reason": "not_bound"}
            if idx is None:
                return {"status": "ignored", "reason": "all_channels_bound"}
            newly_bound = True

        spec = CHANNELS[idx]
        state["last_raw"][signal] = raw
        mode = _resolve_mode(state, signal, raw, message_type)
        if newly_bound or mode is None:
            return {"status": "bound" if newly_bound else "learning", "channel": spec.key, "slot": idx}

        event: Dict[str, Any] = {
            "seq": int(state.get("seq", 0)) + 1,
            "channel": spec.key,
            "slot": idx,
            "signal": signal,
            "mode": mode,
            "raw": raw,
            "timestamp": timestamp,
            "binding_generation": int(state.get("binding_generation", 0)),
        }
        if mode in _RELATIVE_MODES:
            delta = decode_relative(raw, mode)
            if delta == 0:
                return {"status": "ignored", "reason": "zero_relative_delta", "channel": spec.key}
            event["delta"] = delta
        else:
            event["value"] = absolute_to_value(raw, spec)

        q: Deque[Dict[str, Any]] = state["queue"]
        if q.maxlen and len(q) >= q.maxlen:
            state["dropped"] = int(state.get("dropped", 0)) + 1
        q.append(event)
        state["seq"] = event["seq"]
        state["last_channel_event"][spec.key] = deepcopy(event)
        return {"status": "queued", **event}


def drain_events(state: Dict[str, Any], limit: int = 512) -> List[Dict[str, Any]]:
    ensure_transport_state(state)
    out: List[Dict[str, Any]] = []
    with state["lock"]:
        q: Deque[Dict[str, Any]] = state["queue"]
        for _ in range(min(max(0, int(limit)), len(q))):
            out.append(q.popleft())
    return out


def coalesce_events(events: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Reduce a burst to at most one effective event per logical channel.

    Absolute controls keep the latest position. Relative encoders sum their deltas. If the
    mode changes inside a burst (normally only after an explicit override), only events from
    the latest mode segment are retained.
    """
    acc: Dict[str, Dict[str, Any]] = {}
    for raw_event in events:
        event = dict(raw_event)
        channel = str(event.get("channel", ""))
        if not channel:
            continue
        prev = acc.get(channel)
        mode = str(event.get("mode", ""))
        if prev is None or prev.get("mode") != mode:
            acc[channel] = event
            continue
        if mode in _RELATIVE_MODES:
            merged = dict(event)
            merged["delta"] = int(prev.get("delta", 0)) + int(event.get("delta", 0))
            acc[channel] = merged
        else:
            acc[channel] = event
    return sorted(acc.values(), key=lambda e: int(e.get("seq", 0)))


def discard_events(state: Dict[str, Any]) -> int:
    ensure_transport_state(state)
    with state["lock"]:
        n = len(state["queue"])
        state["queue"].clear()
        return n


def discard_events_for_channels(state: Dict[str, Any], channels: Iterable[str]) -> int:
    ensure_transport_state(state)
    wanted = {str(c) for c in channels}
    with state["lock"]:
        before = len(state["queue"])
        state["queue"] = deque(
            (e for e in state["queue"] if str(e.get("channel")) not in wanted),
            maxlen=state["queue"].maxlen,
        )
        return before - len(state["queue"])


def claim_consumer(
    state: Dict[str, Any], owner: str, *, ttl: float = 2.5, force: bool = False,
    now: Optional[float] = None,
) -> bool:
    """Acquire the single-consumer lease for a Streamlit browser session."""
    ensure_transport_state(state)
    now = float(time.time() if now is None else now)
    owner = str(owner)
    with state["lock"]:
        current = state.get("consumer_owner")
        seen = float(state.get("consumer_seen", 0.0) or 0.0)
        stale = not current or now - seen > max(0.5, float(ttl))
        if force or stale or str(current) == owner:
            state["consumer_owner"] = owner
            state["consumer_seen"] = now
            return True
        return False


def touch_consumer(state: Dict[str, Any], owner: str, *, now: Optional[float] = None) -> bool:
    ensure_transport_state(state)
    now = float(time.time() if now is None else now)
    with state["lock"]:
        if str(state.get("consumer_owner")) != str(owner):
            return False
        state["consumer_seen"] = now
        return True


def release_consumer(state: Dict[str, Any], owner: str) -> bool:
    ensure_transport_state(state)
    with state["lock"]:
        if str(state.get("consumer_owner")) != str(owner):
            return False
        state["consumer_owner"] = None
        state["consumer_seen"] = 0.0
        return True


def consumer_is_owner(state: Dict[str, Any], owner: str, *, ttl: float = 2.5,
                      now: Optional[float] = None) -> bool:
    ensure_transport_state(state)
    now = float(time.time() if now is None else now)
    with state["lock"]:
        current = state.get("consumer_owner")
        seen = float(state.get("consumer_seen", 0.0) or 0.0)
        return str(current) == str(owner) and now - seen <= max(0.5, float(ttl))


def set_available_ports(state: Dict[str, Any], ports: Iterable[str]) -> None:
    ensure_transport_state(state)
    cleaned = [str(p) for p in ports if str(p)]
    with state["lock"]:
        state["available_ports"] = list(dict.fromkeys(cleaned))


def set_preferred_port(state: Dict[str, Any], port_name: Optional[str]) -> bool:
    """Set a stable input-port preference and force the listener to reconnect if needed."""
    ensure_transport_state(state)
    preferred = str(port_name).strip() if port_name else None
    to_close = None
    changed = False
    with state["lock"]:
        old = state.get("preferred_port") or None
        changed = old != preferred
        state["preferred_port"] = preferred
        current = state.get("port") or None
        if changed and current and preferred != current:
            state["reconnect_requested"] = True
            to_close = state.get("input_port")
    if to_close is not None:
        try:
            to_close.close()
        except Exception:
            pass
    return changed


def clear_reconnect_request(state: Dict[str, Any]) -> None:
    ensure_transport_state(state)
    with state["lock"]:
        state["reconnect_requested"] = False


def export_config(state: Dict[str, Any]) -> Dict[str, Any]:
    ensure_transport_state(state)
    with state["lock"]:
        return {
            "version": 2,
            "bindings": list(state["bindings"]),
            "mode_overrides": dict(state["mode_overrides"]),
            "auto_bind": bool(state.get("auto_bind", False)),
            "preferred_port": state.get("preferred_port") or None,
        }


def apply_config(state: Dict[str, Any], config: Optional[Dict[str, Any]]) -> None:
    if not isinstance(config, dict):
        return
    ensure_transport_state(state)
    with state["lock"]:
        bindings = list(config.get("bindings") or [])
        if bindings:
            bindings = (bindings + [None] * len(CHANNELS))[:len(CHANNELS)]
            # De-duplicate malformed persisted mappings while preserving first slot.
            seen = set()
            for i, sig in enumerate(bindings):
                if not sig or sig in seen:
                    bindings[i] = None
                else:
                    seen.add(sig)
            state["bindings"] = bindings
            state["mapping"] = [s for s in bindings if s]
        overrides = config.get("mode_overrides")
        if isinstance(overrides, dict):
            state["mode_overrides"] = {}
            for sig, raw_mode in overrides.items():
                mode = "relative_legacy" if str(raw_mode) == "relative" else str(raw_mode)
                if mode == "absolute" or mode in _RELATIVE_MODES:
                    state["mode_overrides"][str(sig)] = mode
            state["cc_modes"].update(state["mode_overrides"])
        state["auto_bind"] = bool(config.get("auto_bind", False))
        preferred = config.get("preferred_port")
        state["preferred_port"] = str(preferred) if preferred else None
        state["learn_slot"] = None
        state["queue"].clear()


def transport_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    ensure_transport_state(state)
    with state["lock"]:
        thread = state.get("listener_thread")
        return {
            "bindings": list(state["bindings"]),
            "cc_modes": dict(state["cc_modes"]),
            "mode_overrides": dict(state["mode_overrides"]),
            "port": state.get("port"),
            "preferred_port": state.get("preferred_port"),
            "available_ports": list(state.get("available_ports") or []),
            "reconnect_requested": bool(state.get("reconnect_requested", False)),
            "events": list(state.get("events") or []),
            "last_raw": dict(state.get("last_raw") or {}),
            "last_channel_event": deepcopy(state.get("last_channel_event") or {}),
            "queued": len(state["queue"]),
            "dropped": int(state.get("dropped", 0)),
            "learn_slot": state.get("learn_slot"),
            "auto_bind": bool(state.get("auto_bind", False)),
            "binding_generation": int(state.get("binding_generation", 0)),
            "consumer_owner": state.get("consumer_owner"),
            "consumer_seen": float(state.get("consumer_seen", 0.0) or 0.0),
            "listener_heartbeat": float(state.get("listener_heartbeat", 0.0) or 0.0),
            "listener_alive": bool(thread and getattr(thread, "is_alive", lambda: False)()),
        }


def new_pickup(target: float, *, armed: bool = True) -> Dict[str, Any]:
    return {"armed": bool(armed), "target": float(target), "last": None}


def apply_absolute_with_pickup(
    current: float,
    incoming: float,
    pickup: Optional[Dict[str, Any]],
    *,
    tolerance: float = 2.0,
) -> Tuple[float, Dict[str, Any], bool]:
    """Apply an absolute hardware value using soft takeover (catch/pickup mode)."""
    p = dict(pickup or new_pickup(current, armed=False))
    incoming = float(incoming)
    if not p.get("armed", False):
        p["last"] = incoming
        return incoming, p, True

    target = float(p.get("target", current))
    last = p.get("last")
    crossed = False
    if last is not None:
        crossed = (float(last) <= target <= incoming) or (incoming <= target <= float(last))
    caught = abs(incoming - target) <= float(tolerance) or crossed
    p["last"] = incoming
    if caught:
        p["armed"] = False
        return incoming, p, True
    return float(current), p, False
