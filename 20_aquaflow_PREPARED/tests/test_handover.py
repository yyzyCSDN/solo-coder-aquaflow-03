import unittest
from datetime import datetime, timezone

from aquaflow.api import AquaFlowService
from aquaflow.operations.handover import (
    HandoverService,
    IsolationWork,
    MissingAsset,
    PressureReading,
    PumpState,
    SafetyEventActive,
    StaleHandover,
    ZoneEvent,
)


def at(hour, minute=0):
    return datetime(2026, 10, 2, hour, minute, tzinfo=timezone.utc)


def build_service():
    """Zone PZ-7: N1 healthy, N2 in low-pressure band, P1 running, P2 stopped."""
    clock = {"now": at(6, 0)}
    svc = HandoverService(clock=lambda: clock["now"])
    svc.register_zone(
        "PZ-7", low_kpa=150, high_kpa=450,
        node_ids={"N1", "N2"}, pump_ids={"P1", "P2"}, valve_ids={"V1", "V2"},
    )
    # corrected = pressure + elevation * 0.980665
    svc.set_readings("PZ-7", [
        PressureReading("N1", elevation_m=10.0, pressure_kpa=300.0),  # 309.8 normal
        PressureReading("N2", elevation_m=50.0, pressure_kpa=100.0),  # 149.0 low
    ])
    svc.set_pumps("PZ-7", [
        PumpState("P1", running=True, kw=45.0, flow=120.0),
        PumpState("P2", running=False, kw=0.0, flow=0.0, note="maintenance"),
    ])
    svc.set_events("PZ-7", [ZoneEvent("EV-9", "customer complaint: discoloured water")])
    svc.set_isolation("PZ-7", [IsolationWork("WO-3", ("V1",), "replace hydrant")])
    return svc, clock


class NormalHandover(unittest.TestCase):
    def test_checkout_and_receive(self):
        svc, clock = build_service()

        record = svc.checkout("PZ-7", "outgoing-ali")
        self.assertEqual(record.version, 4)  # four state edits since registration
        self.assertEqual(record.signed_out_by, "outgoing-ali")

        # Snapshot carries zone pressure, elevation correction, pumps, events, work
        by_node = {s.node_id: s for s in record.pressures}
        self.assertAlmostEqual(by_node["N1"].corrected_kpa, 309.80665, places=3)
        self.assertEqual(by_node["N1"].condition, "normal")
        self.assertEqual(by_node["N2"].condition, "low")
        self.assertEqual(record.unresolved_low, ("N2",))
        self.assertEqual(record.unresolved_high, ())
        pumps = {s.pump_id: s for s in record.pumps}
        self.assertTrue(pumps["P1"].running)
        self.assertFalse(pumps["P2"].running)
        self.assertEqual(pumps["P2"].note, "maintenance")
        self.assertEqual([e.event_id for e in record.open_events], ["EV-9"])
        self.assertEqual(record.pending_isolation[0].work_id, "WO-3")
        self.assertEqual(record.pending_isolation[0].valves, ("V1",))

        clock["now"] = at(14, 0)
        receipt = svc.receive("PZ-7", "incoming-bo", record)
        self.assertEqual(receipt.received_by, "incoming-bo")
        self.assertIs(receipt.record, record)
        self.assertIsNone(svc.current_view("PZ-7")["pending_handover"])

        actions = [(e.action, e.operator, e.reason) for e in svc.audit]
        self.assertEqual(actions, [
            ("checkout", "outgoing-ali", "signed_out"),
            ("receive", "incoming-bo", "received"),
        ])
        self.assertEqual([e.version for e in svc.audit], [4, 4])

    def test_console_view_is_readable(self):
        svc, _ = build_service()
        svc.checkout("PZ-7", "outgoing-ali")
        text = svc.console_view("PZ-7")
        for fragment in ("PRESSURE ZONE PZ-7", "Pump group", "P1", "RUNNING",
                         "Open events", "EV-9", "Pending isolation", "WO-3",
                         "low=N2", "checked out by outgoing-ali"):
            self.assertIn(fragment, text)


class StaleHandoverAfterConcurrentEdit(unittest.TestCase):
    def test_stale_version_refused_with_unresolved_conditions(self):
        svc, clock = build_service()
        record = svc.checkout("PZ-7", "outgoing-ali")

        # Day shift corrects the low-pressure node, but P2 startup drives N1 high
        svc.set_readings("PZ-7", [
            PressureReading("N1", elevation_m=10.0, pressure_kpa=460.0),  # 469.8 high
            PressureReading("N2", elevation_m=50.0, pressure_kpa=160.0),  # 209.0 normal
        ])
        self.assertEqual(svc.current_version("PZ-7"), record.version + 1)

        with self.assertRaises(StaleHandover) as ctx:
            svc.receive("PZ-7", "incoming-bo", record)
        err = ctx.exception
        self.assertEqual(err.record_version, record.version)
        self.assertEqual(err.current_version, record.version + 1)
        self.assertEqual(err.unresolved_high, ["N1"])
        self.assertEqual(err.unresolved_low, [])
        # Message names the still-unresolved condition explicitly
        self.assertIn("stale handover", str(err))
        self.assertIn("high pressure at N1", str(err))

        reject_entries = [e for e in svc.audit if e.action == "reject"]
        self.assertEqual(len(reject_entries), 1)
        self.assertEqual(reject_entries[0].reason, "stale_version")
        self.assertEqual(reject_entries[0].detail["unresolved_high"], ["N1"])
        self.assertIsNone(svc.current_view("PZ-7")["pending_handover"])

    def test_stale_refusal_reports_low_pressure_too(self):
        svc, _ = build_service()
        record = svc.checkout("PZ-7", "outgoing-ali")
        svc.set_readings("PZ-7", [
            PressureReading("N1", elevation_m=10.0, pressure_kpa=100.0),  # low
            PressureReading("N2", elevation_m=50.0, pressure_kpa=90.0),   # low
        ])
        with self.assertRaises(StaleHandover) as ctx:
            svc.receive("PZ-7", "incoming-bo", record)
        self.assertEqual(ctx.exception.unresolved_low, ["N1", "N2"])
        self.assertIn("low pressure at N1,N2", str(ctx.exception))

    def test_explicit_reject_is_audited(self):
        svc, _ = build_service()
        record = svc.checkout("PZ-7", "outgoing-ali")
        svc.reject("PZ-7", "incoming-bo", record, "figures disputed")
        last = svc.audit[-1]
        self.assertEqual((last.action, last.reason), ("reject", "figures disputed"))
        self.assertIsNone(svc.current_view("PZ-7")["pending_handover"])


class BlockedBySafetyEvent(unittest.TestCase):
    def test_checkout_blocked_while_safety_event_active(self):
        svc, _ = build_service()
        svc.set_events("PZ-7", [
            ZoneEvent("EV-9", "customer complaint"),
            ZoneEvent("EV-10", "chlorine residual below action level", safety=True),
        ])
        with self.assertRaises(SafetyEventActive) as ctx:
            svc.checkout("PZ-7", "outgoing-ali")
        self.assertEqual(ctx.exception.active_safety_events, ["EV-10"])
        self.assertIn("EV-10", str(ctx.exception))
        # Blocked sign-out is still recorded as a rejection
        self.assertEqual(svc.audit[-1].action, "reject")
        self.assertEqual(svc.audit[-1].reason, "safety_event_active")
        self.assertIsNone(svc.current_view("PZ-7")["pending_handover"])

    def test_receive_blocked_if_safety_event_opens_between(self):
        svc, _ = build_service()
        record = svc.checkout("PZ-7", "outgoing-ali")
        svc.set_events("PZ-7", [ZoneEvent("EV-11", "gas alarm at pump house", safety=True)])
        with self.assertRaises(SafetyEventActive):
            svc.receive("PZ-7", "incoming-bo", record)
        reasons = [e.reason for e in svc.audit]
        self.assertEqual(reasons[-1], "safety_event_active")


class MissingAssetValidation(unittest.TestCase):
    def test_checkout_rejects_unknown_assets(self):
        svc, _ = build_service()
        svc.set_pumps("PZ-7", [PumpState("P9", running=True)])
        svc.set_isolation("PZ-7", [IsolationWork("WO-4", ("V9",))])
        with self.assertRaises(MissingAsset) as ctx:
            svc.checkout("PZ-7", "outgoing-ali")
        self.assertEqual(ctx.exception.missing["pumps"], ["P9"])
        self.assertEqual(ctx.exception.missing["valves"], ["V9"])
        self.assertIn("pumps=P9", str(ctx.exception))
        self.assertEqual(svc.audit[-1].reason, "missing_assets")

    def test_unknown_zone(self):
        svc, _ = build_service()
        with self.assertRaises(ValueError):
            svc.current_view("NOPE")


class FacadeWiring(unittest.TestCase):
    def test_service_exposes_handovers(self):
        svc = AquaFlowService()
        self.assertIsInstance(svc.handovers, HandoverService)
        self.assertEqual(svc.health()["status"], "ok")


if __name__ == "__main__":
    unittest.main()
