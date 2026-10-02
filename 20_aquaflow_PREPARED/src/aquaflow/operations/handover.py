"""Shift handover for pressure zone operators.

A handover bundles the information an outgoing operator must leave for the
incoming crew: the current (elevation corrected) zone pressure, the elevation
corrections behind it, pump group status, still-open events and pending
isolation work.  The outgoing operator signs out, producing an immutable
record stamped with the zone revision.  The incoming operator either accepts
that record or rejects it.

Optimistic concurrency: every domain mutation bumps the zone revision.  If the
zone changed after a record was signed out, accepting the stale record is
refused and the rejection spells out every low/high pressure condition that is
still unresolved.  A handover is also blocked outright while a security event
is active, so command responsibility is never transferred during an active
security response.

Every sign out, acceptance and rejection appends an audit entry.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
from datetime import datetime

from ..network.pressure import Node as _PressureNode, classify as _classify_pressure

GRAVITY_KPA_PER_M = 9.80665 / 10.0


class HandoverError(ValueError):
    """Base class for handover failures."""


class MissingAssetError(HandoverError):
    """A handover references assets absent from the zone asset registry."""


class HandoverBlockedError(HandoverError):
    """A handover cannot proceed, e.g. because a security event is active."""


class StaleHandoverError(HandoverError):
    """The presented record is older than the current zone revision."""

    def __init__(self, message: str, expected_revision: int, actual_revision: int):
        super().__init__(message)
        self.expected_revision = expected_revision
        self.actual_revision = actual_revision
        self.low_pressure: tuple[str, ...] = ()
        self.high_pressure: tuple[str, ...] = ()


@dataclass(frozen=True)
class PressurePoint:
    """A pressure monitor and its operating envelope."""

    id: str
    elevation_m: float
    pressure_kpa: float
    low_limit_kpa: float
    high_limit_kpa: float


@dataclass(frozen=True)
class PumpStatus:
    """State of one pump in the zone's pump group."""

    id: str
    running: bool
    healthy: bool = True
    kw: float = 0.0


@dataclass(frozen=True)
class ZoneEvent:
    """An operational event.  Closed events are excluded from a handover."""

    id: str
    kind: str
    closed: bool = False


@dataclass(frozen=True)
class IsolationWork:
    """Pending valve isolation work handed to the incoming operator."""

    work_order: str
    valves: frozenset[str]
    completed: bool = False


@dataclass(frozen=True)
class HandoverRecord:
    """Immutable snapshot signed out by the outgoing operator."""

    zone_id: str
    revision: int
    signed_out_by: str
    signed_out_at: datetime
    pressure: dict[str, object]
    elevation_correction_kpa: dict[str, float]
    pumps: tuple[PumpStatus, ...]
    open_events: tuple[ZoneEvent, ...]
    pending_isolation: tuple[IsolationWork, ...]
    accepted_by: str | None = None
    accepted_at: datetime | None = None


@dataclass(frozen=True)
class AuditEntry:
    """One appended audit fact for a zone."""

    seq: int
    zone_id: str
    action: str
    at: datetime
    actor: str
    revision: int
    detail: str = ""


@dataclass(frozen=True)
class ZoneView:
    """Console-friendly snapshot of the current zone state."""

    zone_id: str
    revision: int
    controller: str
    signed_out: bool
    pressure: dict[str, object]
    pumps: tuple[PumpStatus, ...]
    open_events: tuple[ZoneEvent, ...]
    pending_isolation: tuple[IsolationWork, ...]
    missing_assets: tuple[str, ...]
    lines: tuple[str, ...]

    def render(self) -> str:
        return "\n".join(self.lines)


@dataclass
class _Zone:
    controller: str
    points: tuple[PressurePoint, ...]
    pumps: tuple[PumpStatus, ...]
    events: tuple[ZoneEvent, ...]
    isolation: tuple[IsolationWork, ...]
    assets: frozenset[str]
    revision: int = 0
    pending: HandoverRecord | None = None


def corrected_kpa(point: PressurePoint) -> float:
    """Pressure corrected to the zone datum elevation (kPa)."""
    return point.pressure_kpa + point.elevation_m * GRAVITY_KPA_PER_M


def _pressure_snapshot(points: tuple[PressurePoint, ...]) -> dict[str, object]:
    nodes = [_PressureNode(p.id, p.elevation_m, p.pressure_kpa) for p in points]
    if points:
        low = min(p.low_limit_kpa for p in points)
        high = max(p.high_limit_kpa for p in points)
        result = _classify_pressure(nodes, low, high)
    else:
        result = {"low": [], "high": []}
    readings = {
        p.id: {
            "pressure_kpa": p.pressure_kpa,
            "elevation_m": p.elevation_m,
            "corrected_kpa": corrected_kpa(p),
            "low_limit_kpa": p.low_limit_kpa,
            "high_limit_kpa": p.high_limit_kpa,
        }
        for p in points
    }
    return {
        "low": tuple(result["low"]),
        "high": tuple(result["high"]),
        "readings": readings,
    }


def _missing_assets(zone: _Zone) -> tuple[str, ...]:
    referenced: set[str] = {p.id for p in zone.points}
    referenced.update(pump.id for pump in zone.pumps)
    for work in zone.isolation:
        referenced.update(work.valves)
    return tuple(sorted(referenced - zone.assets))


def _open_events(zone: _Zone) -> tuple[ZoneEvent, ...]:
    return tuple(e for e in zone.events if not e.closed)


def _pending_isolation(zone: _Zone) -> tuple[IsolationWork, ...]:
    return tuple(w for w in zone.isolation if not w.completed)


def _security_event_open(zone: _Zone) -> ZoneEvent | None:
    return next((e for e in zone.events if not e.closed and e.kind == "security"), None)


def _pressure_reasons(zone: _Zone) -> tuple[tuple[str, ...], tuple[str, ...]]:
    snapshot = _pressure_snapshot(zone.points)
    readings = snapshot["readings"]
    low = tuple(
        f"{pid} corrected={readings[pid]['corrected_kpa']:.2f}kPa"
        f" below low_limit={readings[pid]['low_limit_kpa']:.2f}kPa"
        for pid in snapshot["low"]
    )
    high = tuple(
        f"{pid} corrected={readings[pid]['corrected_kpa']:.2f}kPa"
        f" above high_limit={readings[pid]['high_limit_kpa']:.2f}kPa"
        for pid in snapshot["high"]
    )
    return low, high


class HandoverService:
    """Stateful, in-memory handover ledger for pressure zones."""

    def __init__(self) -> None:
        self._zones: dict[str, _Zone] = {}
        self._audit: list[AuditEntry] = []
        self._seq = 0

    # ------------------------------------------------------------------ setup
    def register_zone(
        self,
        zone_id: str,
        controller: str,
        points: list[PressurePoint] | tuple[PressurePoint, ...] = (),
        pumps: list[PumpStatus] | tuple[PumpStatus, ...] = (),
        events: list[ZoneEvent] | tuple[ZoneEvent, ...] = (),
        isolation: list[IsolationWork] | tuple[IsolationWork, ...] = (),
        assets: frozenset[str] | set[str] | tuple[str, ...] | None = None,
    ) -> None:
        if not zone_id:
            raise ValueError("zone id required")
        if not controller:
            raise ValueError("controller required")
        if assets is None:
            registry = {p.id for p in points} | {pump.id for pump in pumps}
            for work in isolation:
                registry |= set(work.valves)
            assets = frozenset(registry)
        self._zones[zone_id] = _Zone(
            controller=controller,
            points=tuple(points),
            pumps=tuple(pumps),
            events=tuple(events),
            isolation=tuple(isolation),
            assets=frozenset(assets),
        )

    def _zone(self, zone_id: str) -> _Zone:
        try:
            return self._zones[zone_id]
        except KeyError:
            raise KeyError(f"unknown zone: {zone_id}") from None

    # --------------------------------------------------------------- mutation
    def _bump(self, zone: _Zone) -> None:
        zone.revision += 1

    def set_pressure(self, zone_id: str, point_id: str, pressure_kpa: float) -> None:
        zone = self._zone(zone_id)
        if not any(p.id == point_id for p in zone.points):
            raise KeyError(f"unknown pressure point: {point_id}")
        zone.points = tuple(
            replace(p, pressure_kpa=float(pressure_kpa)) if p.id == point_id else p
            for p in zone.points
        )
        self._bump(zone)

    def set_pump(self, zone_id: str, pump_id: str, running: bool, healthy: bool | None = None) -> None:
        zone = self._zone(zone_id)
        if not any(p.id == pump_id for p in zone.pumps):
            raise KeyError(f"unknown pump: {pump_id}")
        zone.pumps = tuple(
            replace(p, running=running, healthy=p.healthy if healthy is None else healthy)
            if p.id == pump_id
            else p
            for p in zone.pumps
        )
        self._bump(zone)

    def open_event(self, zone_id: str, event: ZoneEvent) -> None:
        zone = self._zone(zone_id)
        if any(e.id == event.id for e in zone.events):
            raise ValueError(f"duplicate event: {event.id}")
        zone.events = (*zone.events, event)
        self._bump(zone)

    def close_event(self, zone_id: str, event_id: str) -> None:
        zone = self._zone(zone_id)
        if not any(e.id == event_id for e in zone.events):
            raise KeyError(f"unknown event: {event_id}")
        zone.events = tuple(
            replace(e, closed=True) if e.id == event_id else e for e in zone.events
        )
        self._bump(zone)

    def complete_isolation(self, zone_id: str, work_order: str) -> None:
        zone = self._zone(zone_id)
        if not any(w.work_order == work_order for w in zone.isolation):
            raise KeyError(f"unknown isolation work: {work_order}")
        zone.isolation = tuple(
            replace(w, completed=True) if w.work_order == work_order else w
            for w in zone.isolation
        )
        self._bump(zone)

    # ------------------------------------------------------------------ audit
    def _append_audit(
        self,
        zone_id: str,
        action: str,
        actor: str,
        revision: int,
        detail: str,
        at: datetime | None,
    ) -> AuditEntry:
        self._seq += 1
        entry = AuditEntry(
            seq=self._seq,
            zone_id=zone_id,
            action=action,
            at=at or datetime.now().astimezone(),
            actor=actor,
            revision=revision,
            detail=detail,
        )
        self._audit.append(entry)
        return entry

    def audit_log(self, zone_id: str | None = None) -> tuple[AuditEntry, ...]:
        if zone_id is None:
            return tuple(self._audit)
        return tuple(e for e in self._audit if e.zone_id == zone_id)

    # -------------------------------------------------------------- handover
    def sign_out(
        self, zone_id: str, outgoing_operator: str, at: datetime | None = None
    ) -> HandoverRecord:
        """Outgoing operator signs out and produces a pending handover record."""
        if not outgoing_operator:
            raise ValueError("outgoing operator required")
        zone = self._zone(zone_id)
        if zone.pending is not None:
            raise HandoverBlockedError(
                f"zone {zone_id} already has a pending handover from "
                f"{zone.pending.signed_out_by} at revision {zone.pending.revision}"
            )
        missing = _missing_assets(zone)
        if missing:
            raise MissingAssetError(
                "missing assets referenced by zone state: " + ",".join(missing)
            )
        active_security = _security_event_open(zone)
        if active_security is not None:
            error = HandoverBlockedError(
                f"handover blocked for zone {zone_id}: security event "
                f"{active_security.id} is still active"
            )
            self._append_audit(
                zone_id, "reject", outgoing_operator, zone.revision, str(error), at
            )
            raise error

        pressure = _pressure_snapshot(zone.points)
        record = HandoverRecord(
            zone_id=zone_id,
            revision=zone.revision,
            signed_out_by=outgoing_operator,
            signed_out_at=at or datetime.now().astimezone(),
            pressure=pressure,
            elevation_correction_kpa={
                p.id: round(p.elevation_m * GRAVITY_KPA_PER_M, 6) for p in zone.points
            },
            pumps=zone.pumps,
            open_events=_open_events(zone),
            pending_isolation=_pending_isolation(zone),
        )
        zone.pending = record
        self._append_audit(
            zone_id,
            "sign_out",
            outgoing_operator,
            zone.revision,
            f"revision={zone.revision} open_events={len(record.open_events)} "
            f"pending_isolation={len(record.pending_isolation)} "
            f"low={','.join(pressure['low']) or '-'} high={','.join(pressure['high']) or '-'}",
            at,
        )
        return record

    def _refuse(
        self,
        zone: _Zone,
        record: HandoverRecord,
        actor: str,
        reason: str,
        at: datetime | None,
        low: tuple[str, ...] = (),
        high: tuple[str, ...] = (),
    ) -> StaleHandoverError:
        zone.pending = None
        self._append_audit(record.zone_id, "reject", actor, zone.revision, reason, at)
        error = StaleHandoverError(reason, record.revision, zone.revision)
        error.low_pressure = low
        error.high_pressure = high
        return error

    def accept(
        self,
        zone_id: str,
        incoming_operator: str,
        record: HandoverRecord,
        at: datetime | None = None,
    ) -> HandoverRecord:
        """Incoming operator receives a signed-out record."""
        if not incoming_operator:
            raise ValueError("incoming operator required")
        zone = self._zone(zone_id)
        if record.zone_id != zone_id:
            raise ValueError("record belongs to a different zone")
        if incoming_operator == record.signed_out_by:
            raise ValueError("incoming operator must differ from outgoing operator")
        if zone.pending is None:
            raise HandoverError(f"no pending handover for zone {zone_id}")
        if zone.pending is not record:
            raise HandoverError("record is not the current pending handover")

        low, high = _pressure_reasons(zone)
        security = _security_event_open(zone)
        if security is not None:
            zone.pending = None
            reason = (
                f"handover blocked for zone {zone_id}: security event "
                f"{security.id} is still active"
            )
            self._append_audit(zone_id, "reject", incoming_operator, zone.revision, reason, at)
            raise HandoverBlockedError(reason)
        if record.revision != zone.revision:
            parts = [
                f"stale handover rejected for zone {zone_id}: record revision "
                f"{record.revision} is older than current revision {zone.revision}"
            ]
            if low:
                parts.append("unresolved low pressure: " + "; ".join(low))
            if high:
                parts.append("unresolved high pressure: " + "; ".join(high))
            if not low and not high:
                parts.append("no low/high pressure conditions currently open")
            reason = ". ".join(parts)
            raise self._refuse(zone, record, incoming_operator, reason, at, low, high)

        accepted = replace(
            record,
            accepted_by=incoming_operator,
            accepted_at=at or datetime.now().astimezone(),
        )
        zone.pending = None
        zone.controller = incoming_operator
        self._append_audit(
            zone_id,
            "accept",
            incoming_operator,
            zone.revision,
            f"revision={zone.revision} from={record.signed_out_by}",
            at,
        )
        return accepted

    def reject(
        self,
        zone_id: str,
        incoming_operator: str,
        record: HandoverRecord,
        at: datetime | None = None,
    ) -> None:
        """Incoming operator explicitly refuses a pending record."""
        if not incoming_operator:
            raise ValueError("incoming operator required")
        zone = self._zone(zone_id)
        if record.zone_id != zone_id:
            raise ValueError("record belongs to a different zone")
        if zone.pending is None or zone.pending is not record:
            raise HandoverError("record is not the current pending handover")
        low, high = _pressure_reasons(zone)
        parts = [
            f"handover rejected for zone {zone_id} by {incoming_operator}",
            f"record_revision={record.revision} current_revision={zone.revision}",
        ]
        if low:
            parts.append("unresolved low pressure: " + "; ".join(low))
        if high:
            parts.append("unresolved high pressure: " + "; ".join(high))
        security = _security_event_open(zone)
        if security is not None:
            parts.append(f"active security event: {security.id}")
        reason = ". ".join(parts)
        zone.pending = None
        self._append_audit(zone_id, "reject", incoming_operator, zone.revision, reason, at)

    # -------------------------------------------------------------------- view
    def current_view(self, zone_id: str) -> ZoneView:
        zone = self._zone(zone_id)
        pressure = _pressure_snapshot(zone.points)
        view = ZoneView(
            zone_id=zone_id,
            revision=zone.revision,
            controller=zone.controller,
            signed_out=zone.pending is not None,
            pressure=pressure,
            pumps=zone.pumps,
            open_events=_open_events(zone),
            pending_isolation=_pending_isolation(zone),
            missing_assets=_missing_assets(zone),
            lines=(),
        )
        lines = _render(view, zone.pending)
        return replace(view, lines=tuple(lines))


def _pump_label(pump: PumpStatus) -> str:
    if not pump.healthy:
        return "FAULT"
    return "RUN" if pump.running else "STOP"


def _render(view: ZoneView, pending: HandoverRecord | None) -> list[str]:
    status = "SIGNED OUT (awaiting acceptance)" if view.signed_out else "in service"
    lines = [
        f"PRESSURE ZONE {view.zone_id}  revision={view.revision}  [{status}]",
        f"controller: {view.controller}",
    ]
    readings = view.pressure["readings"]
    if readings:
        lines.append("pressure (elevation corrected):")
        for pid in sorted(readings):
            r = readings[pid]
            if pid in view.pressure["low"]:
                marker = "LOW "
            elif pid in view.pressure["high"]:
                marker = "HIGH"
            else:
                marker = "OK  "
            lines.append(
                f"  [{marker}] {pid:<12} raw={r['pressure_kpa']:>8.2f} kPa"
                f"  elev={r['elevation_m']:>7.2f} m"
                f"  corrected={r['corrected_kpa']:>8.2f} kPa"
                f"  envelope=[{r['low_limit_kpa']:.2f},{r['high_limit_kpa']:.2f}]"
            )
    else:
        lines.append("pressure (elevation corrected): no points")
    if view.pumps:
        lines.append("pump group:")
        for pump in sorted(view.pumps, key=lambda p: p.id):
            lines.append(
                f"  [{_pump_label(pump)}] {pump.id:<12} kw={pump.kw:.2f}"
            )
    else:
        lines.append("pump group: none")
    if view.open_events:
        lines.append("open events:")
        for event in sorted(view.open_events, key=lambda e: e.id):
            lines.append(f"  [OPEN] {event.id:<12} kind={event.kind}")
    else:
        lines.append("open events: none")
    if view.pending_isolation:
        lines.append("pending isolation work:")
        for work in sorted(view.pending_isolation, key=lambda w: w.work_order):
            valves = ",".join(sorted(work.valves)) or "-"
            lines.append(f"  [PEND] {work.work_order:<12} valves={valves}")
    else:
        lines.append("pending isolation work: none")
    if view.missing_assets:
        lines.append("MISSING ASSETS: " + ",".join(view.missing_assets))
    if pending is not None:
        lines.append(
            f"pending handover: signed_out_by={pending.signed_out_by}"
            f" at={pending.signed_out_at.isoformat()} record_revision={pending.revision}"
        )
    return lines
