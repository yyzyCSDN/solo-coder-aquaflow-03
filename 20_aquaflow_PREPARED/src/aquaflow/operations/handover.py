"""Shift handover for pressure-zone operators.

A handover is an optimistic-concurrency snapshot: the outgoing operator signs
out the current zone state (pressures with elevation correction, pump group
state, open events and pending isolation work), and the incoming operator
later receives that exact version.  Any concurrent change bumps the zone
version, so a stale snapshot is refused and the refusal names the low/high
pressure conditions still unresolved.  Active safety events block a handover,
referenced assets are validated against the zone registry, and every
checkout / receive / reject is appended to an audit trail.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Callable

# Elevation correction shared with aquaflow.network.pressure.classify:
# corrected_kpa = pressure_kpa + elevation_m * g / 10
_GRAVITY_KPA_PER_M = 9.80665 / 10


def corrected_pressure(elevation_m: float, pressure_kpa: float) -> float:
    return pressure_kpa + elevation_m * _GRAVITY_KPA_PER_M


class HandoverError(ValueError):
    """Base class for refused pressure-zone handovers."""


class ZoneNotFound(HandoverError):
    def __init__(self, zone_id: str):
        super().__init__(f"unknown pressure zone: {zone_id}")
        self.zone_id = zone_id


class MissingAsset(HandoverError):
    """Sign-out references assets absent from the zone registry."""

    def __init__(self, zone_id: str, missing: dict[str, list[str]]):
        self.zone_id = zone_id
        self.missing = {kind: sorted(ids) for kind, ids in missing.items() if ids}
        detail = "; ".join(f"{kind}={','.join(ids)}" for kind, ids in sorted(self.missing.items()))
        super().__init__(f"zone {zone_id} references missing assets: {detail}")


class SafetyEventActive(HandoverError):
    """A handover cannot proceed while safety events remain open."""

    def __init__(self, zone_id: str, active_safety_events: list[str]):
        self.zone_id = zone_id
        self.active_safety_events = sorted(active_safety_events)
        super().__init__(
            f"zone {zone_id} handover blocked by active safety events: "
            f"{','.join(self.active_safety_events)}"
        )


class StaleHandover(HandoverError):
    """The presented record is older than the current zone version."""

    def __init__(self, zone_id: str, record_version: int, current_version: int,
                 unresolved_low: list[str], unresolved_high: list[str]):
        self.zone_id = zone_id
        self.record_version = record_version
        self.current_version = current_version
        self.unresolved_low = sorted(unresolved_low)
        self.unresolved_high = sorted(unresolved_high)
        conditions = []
        if self.unresolved_low:
            conditions.append("low pressure at " + ",".join(self.unresolved_low))
        if self.unresolved_high:
            conditions.append("high pressure at " + ",".join(self.unresolved_high))
        condition_text = "; ".join(conditions) if conditions else "no unresolved low/high pressure condition"
        super().__init__(
            f"zone {zone_id} stale handover v{record_version}, current is v{current_version}; "
            f"refused. Still unresolved -> {condition_text}"
        )


@dataclass(frozen=True)
class PressureReading:
    node_id: str
    elevation_m: float
    pressure_kpa: float


@dataclass(frozen=True)
class PumpState:
    pump_id: str
    running: bool
    kw: float = 0.0
    flow: float = 0.0
    note: str = ""


@dataclass(frozen=True)
class ZoneEvent:
    """An open operational event; safety events block handover."""

    event_id: str
    summary: str
    safety: bool = False


@dataclass(frozen=True)
class IsolationWork:
    work_id: str
    valves: tuple[str, ...]
    note: str = ""


@dataclass(frozen=True)
class PressureSnapshot:
    node_id: str
    elevation_m: float
    pressure_kpa: float
    corrected_kpa: float
    condition: str  # low | normal | high


@dataclass(frozen=True)
class PumpSnapshot:
    pump_id: str
    running: bool
    kw: float
    flow: float
    note: str


@dataclass(frozen=True)
class EventSnapshot:
    event_id: str
    summary: str
    safety: bool


@dataclass(frozen=True)
class IsolationSnapshot:
    work_id: str
    valves: tuple[str, ...]
    note: str


@dataclass(frozen=True)
class HandoverRecord:
    """Immutable snapshot signed out by the outgoing operator."""

    zone_id: str
    version: int
    low_kpa: float
    high_kpa: float
    pressures: tuple[PressureSnapshot, ...]
    pumps: tuple[PumpSnapshot, ...]
    open_events: tuple[EventSnapshot, ...]
    pending_isolation: tuple[IsolationSnapshot, ...]
    unresolved_low: tuple[str, ...]
    unresolved_high: tuple[str, ...]
    signed_out_by: str
    signed_out_at: datetime


@dataclass(frozen=True)
class HandoverReceipt:
    """Incoming operator's acceptance of a handover record."""

    record: HandoverRecord
    received_by: str
    received_at: datetime


@dataclass(frozen=True)
class AuditEntry:
    seq: int
    at: datetime
    action: str  # checkout | receive | reject
    zone_id: str
    version: int
    operator: str
    reason: str
    detail: dict[str, object] = field(default_factory=dict)


@dataclass
class _Zone:
    zone_id: str
    low_kpa: float
    high_kpa: float
    node_ids: set[str]
    pump_ids: set[str]
    valve_ids: set[str]
    readings: list[PressureReading] = field(default_factory=list)
    pumps: list[PumpState] = field(default_factory=list)
    events: list[ZoneEvent] = field(default_factory=list)
    isolation: list[IsolationWork] = field(default_factory=list)
    version: int = 0
    pending: HandoverRecord | None = None


def _pressure_snapshots(readings: list[PressureReading], low: float, high: float) -> tuple[PressureSnapshot, ...]:
    rows = []
    for r in readings:
        corrected = corrected_pressure(r.elevation_m, r.pressure_kpa)
        if corrected < low:
            condition = "low"
        elif corrected > high:
            condition = "high"
        else:
            condition = "normal"
        rows.append(PressureSnapshot(r.node_id, r.elevation_m, r.pressure_kpa, corrected, condition))
    return tuple(sorted(rows, key=lambda s: s.node_id))


def _unresolved(snapshots: tuple[PressureSnapshot, ...]) -> tuple[tuple[str, ...], tuple[str, ...]]:
    low = tuple(s.node_id for s in snapshots if s.condition == "low")
    high = tuple(s.node_id for s in snapshots if s.condition == "high")
    return low, high


class HandoverService:
    """Register pressure zones and run sign-out / receive handovers."""

    def __init__(self, clock: Callable[[], datetime] | None = None):
        self._zones: dict[str, _Zone] = {}
        self._audit: list[AuditEntry] = []
        self._clock = clock or (lambda: datetime.now().astimezone())

    # ------------------------------------------------------------------ setup

    def register_zone(self, zone_id: str, *, low_kpa: float, high_kpa: float,
                      node_ids: set[str] | tuple[str, ...],
                      pump_ids: set[str] | tuple[str, ...],
                      valve_ids: set[str] | tuple[str, ...]) -> None:
        if low_kpa >= high_kpa:
            raise ValueError("invalid pressure envelope")
        self._zones[zone_id] = _Zone(
            zone_id, float(low_kpa), float(high_kpa),
            set(node_ids), set(pump_ids), set(valve_ids),
        )

    def _zone(self, zone_id: str) -> _Zone:
        zone = self._zones.get(zone_id)
        if zone is None:
            raise ZoneNotFound(zone_id)
        return zone

    # ------------------------------------------------- concurrent state edits

    def set_readings(self, zone_id: str, readings: list[PressureReading]) -> None:
        zone = self._zone(zone_id)
        zone.readings = list(readings)
        zone.version += 1

    def set_pumps(self, zone_id: str, pumps: list[PumpState]) -> None:
        zone = self._zone(zone_id)
        zone.pumps = list(pumps)
        zone.version += 1

    def set_events(self, zone_id: str, events: list[ZoneEvent]) -> None:
        zone = self._zone(zone_id)
        zone.events = list(events)
        zone.version += 1

    def set_isolation(self, zone_id: str, works: list[IsolationWork]) -> None:
        zone = self._zone(zone_id)
        zone.isolation = list(works)
        zone.version += 1

    def current_version(self, zone_id: str) -> int:
        return self._zone(zone_id).version

    # ------------------------------------------------------------- validation

    def _missing_assets(self, zone: _Zone) -> dict[str, list[str]]:
        nodes = sorted({r.node_id for r in zone.readings if r.node_id not in zone.node_ids})
        pumps = sorted({p.pump_id for p in zone.pumps if p.pump_id not in zone.pump_ids})
        valves = sorted({v for work in zone.isolation for v in work.valves if v not in zone.valve_ids})
        return {"nodes": nodes, "pumps": pumps, "valves": valves}

    def _active_safety(self, zone: _Zone) -> list[str]:
        return sorted(e.event_id for e in zone.events if e.safety)

    # ---------------------------------------------------------------- checkout

    def checkout(self, zone_id: str, operator: str) -> HandoverRecord:
        """Sign out the current zone state for the incoming shift."""
        zone = self._zone(zone_id)
        missing = self._missing_assets(zone)
        if any(missing.values()):
            err = MissingAsset(zone_id, missing)
            self._append("reject", zone, operator, "missing_assets",
                         {"missing": err.missing})
            raise err
        active_safety = self._active_safety(zone)
        if active_safety:
            err = SafetyEventActive(zone_id, active_safety)
            self._append("reject", zone, operator, "safety_event_active",
                         {"active_safety_events": active_safety})
            raise err

        snapshots = _pressure_snapshots(zone.readings, zone.low_kpa, zone.high_kpa)
        low, high = _unresolved(snapshots)
        record = HandoverRecord(
            zone_id=zone.zone_id,
            version=zone.version,
            low_kpa=zone.low_kpa,
            high_kpa=zone.high_kpa,
            pressures=snapshots,
            pumps=tuple(sorted(
                (PumpSnapshot(p.pump_id, p.running, p.kw, p.flow, p.note) for p in zone.pumps),
                key=lambda s: s.pump_id,
            )),
            open_events=tuple(sorted(
                (EventSnapshot(e.event_id, e.summary, e.safety) for e in zone.events),
                key=lambda s: s.event_id,
            )),
            pending_isolation=tuple(sorted(
                (IsolationSnapshot(w.work_id, tuple(w.valves), w.note) for w in zone.isolation),
                key=lambda s: s.work_id,
            )),
            unresolved_low=low,
            unresolved_high=high,
            signed_out_by=operator,
            signed_out_at=self._clock(),
        )
        zone.pending = record
        self._append("checkout", zone, operator, "signed_out",
                     {"record_version": record.version,
                      "unresolved_low": list(low), "unresolved_high": list(high)})
        return record

    # ----------------------------------------------------------------- receive

    def receive(self, zone_id: str, operator: str, record: HandoverRecord) -> HandoverReceipt:
        """Accept a signed-out record; refuse stale or blocked versions."""
        zone = self._zone(zone_id)
        if record.zone_id != zone.zone_id:
            raise ValueError("record does not belong to zone " + zone.zone_id)

        missing = self._missing_assets(zone)
        if any(missing.values()):
            err = MissingAsset(zone_id, missing)
            self._append("reject", zone, operator, "missing_assets",
                         {"record_version": record.version, "missing": err.missing})
            raise err

        active_safety = self._active_safety(zone)
        if active_safety:
            err = SafetyEventActive(zone_id, active_safety)
            self._append("reject", zone, operator, "safety_event_active",
                         {"record_version": record.version,
                          "active_safety_events": active_safety})
            raise err

        if record.version != zone.version:
            snapshots = _pressure_snapshots(zone.readings, zone.low_kpa, zone.high_kpa)
            low, high = _unresolved(snapshots)
            err = StaleHandover(zone_id, record.version, zone.version, list(low), list(high))
            zone.pending = None
            self._append("reject", zone, operator, "stale_version",
                         {"record_version": record.version, "current_version": zone.version,
                          "unresolved_low": list(low), "unresolved_high": list(high)})
            raise err

        receipt = HandoverReceipt(record=record, received_by=operator, received_at=self._clock())
        zone.pending = None
        self._append("receive", zone, operator, "received",
                     {"record_version": record.version, "signed_out_by": record.signed_out_by})
        return receipt

    def reject(self, zone_id: str, operator: str, record: HandoverRecord, reason: str) -> None:
        """Incoming operator explicitly refuses a pending record."""
        zone = self._zone(zone_id)
        if record.zone_id != zone.zone_id:
            raise ValueError("record does not belong to zone " + zone.zone_id)
        snapshots = _pressure_snapshots(zone.readings, zone.low_kpa, zone.high_kpa)
        low, high = _unresolved(snapshots)
        zone.pending = None
        self._append("reject", zone, operator, reason or "rejected",
                     {"record_version": record.version, "current_version": zone.version,
                      "stale": record.version != zone.version,
                      "unresolved_low": list(low), "unresolved_high": list(high)})

    # ------------------------------------------------------------------- views

    def current_view(self, zone_id: str) -> dict[str, object]:
        """Structured current view of the zone."""
        zone = self._zone(zone_id)
        snapshots = _pressure_snapshots(zone.readings, zone.low_kpa, zone.high_kpa)
        low, high = _unresolved(snapshots)
        pending = None
        if zone.pending is not None:
            pending = {
                "version": zone.pending.version,
                "signed_out_by": zone.pending.signed_out_by,
                "signed_out_at": zone.pending.signed_out_at.isoformat(),
            }
        return {
            "zone_id": zone.zone_id,
            "version": zone.version,
            "low_kpa": zone.low_kpa,
            "high_kpa": zone.high_kpa,
            "pressures": [s.__dict__ for s in snapshots],
            "pumps": [s.__dict__ for s in sorted(
                (PumpSnapshot(p.pump_id, p.running, p.kw, p.flow, p.note) for p in zone.pumps),
                key=lambda s: s.pump_id)],
            "open_events": [s.__dict__ for s in sorted(
                (EventSnapshot(e.event_id, e.summary, e.safety) for e in zone.events),
                key=lambda s: s.event_id)],
            "pending_isolation": [{"work_id": w.work_id, "valves": list(w.valves), "note": w.note}
                                  for w in sorted(zone.isolation, key=lambda w: w.work_id)],
            "unresolved_low": list(low),
            "unresolved_high": list(high),
            "active_safety_events": self._active_safety(zone),
            "missing_assets": {k: v for k, v in self._missing_assets(zone).items() if v},
            "pending_handover": pending,
        }

    def console_view(self, zone_id: str) -> str:
        """Human/console-readable rendering of the current zone view."""
        view = self.current_view(zone_id)
        lines = [
            f"PRESSURE ZONE {view['zone_id']} @ v{view['version']}  "
            f"(envelope {view['low_kpa']:.1f}-{view['high_kpa']:.1f} kPa, elevation-corrected)",
            "-" * 68,
        ]
        lines.append("Pressure:")
        if view["pressures"]:
            for s in view["pressures"]:
                marker = {"low": "[LOW]", "high": "[HIGH]", "normal": ""}[s["condition"]]
                lines.append(
                    f"  {s['node_id']:<6} raw {s['pressure_kpa']:8.1f} kPa  "
                    f"elev {s['elevation_m']:6.1f} m  corr {s['corrected_kpa']:8.1f} kPa {marker}"
                )
        else:
            lines.append("  (no readings)")

        lines.append("Pump group:")
        if view["pumps"]:
            for s in view["pumps"]:
                state = "RUNNING" if s["running"] else "STOPPED"
                note = f"  {s['note']}" if s["note"] else ""
                lines.append(f"  {s['pump_id']:<6} {state:<7} kw={s['kw']:.1f} flow={s['flow']:.1f}{note}")
        else:
            lines.append("  (no pumps registered)")

        lines.append(f"Open events ({len(view['open_events'])}):")
        for s in view["open_events"]:
            flag = " [SAFETY]" if s["safety"] else ""
            lines.append(f"  {s['event_id']:<6}{flag} {s['summary']}")
        if not view["open_events"]:
            lines.append("  (none)")

        lines.append(f"Pending isolation work ({len(view['pending_isolation'])}):")
        for w in view["pending_isolation"]:
            valves = ",".join(w["valves"]) if w["valves"] else "-"
            note = f"  {w['note']}" if w["note"] else ""
            lines.append(f"  {w['work_id']:<6} valves={valves}{note}")
        if not view["pending_isolation"]:
            lines.append("  (none)")

        low = ",".join(view["unresolved_low"]) or "-"
        high = ",".join(view["unresolved_high"]) or "-"
        lines.append(f"Unresolved conditions: low={low} high={high}")
        missing = view["missing_assets"]
        if missing:
            detail = "; ".join(f"{k}={','.join(v)}" for k, v in sorted(missing.items()))
            lines.append(f"WARNING missing assets: {detail}")
        safety = view["active_safety_events"]
        if safety:
            lines.append("BLOCKED: active safety event(s): " + ",".join(safety))

        pending = view["pending_handover"]
        if pending:
            lines.append(
                f"Handover: checked out by {pending['signed_out_by']} at "
                f"{pending['signed_out_at']} (record v{pending['version']})"
            )
        else:
            lines.append("Handover: none pending")
        return "\n".join(lines)

    # ------------------------------------------------------------------ audit

    def _append(self, action: str, zone: _Zone, operator: str, reason: str,
                detail: dict[str, object]) -> AuditEntry:
        entry = AuditEntry(
            seq=len(self._audit) + 1,
            at=self._clock(),
            action=action,
            zone_id=zone.zone_id,
            version=zone.version,
            operator=operator,
            reason=reason,
            detail=detail,
        )
        self._audit.append(entry)
        return entry

    @property
    def audit(self) -> tuple[AuditEntry, ...]:
        return tuple(self._audit)
