import unittest
from datetime import datetime, timezone

from aquaflow.operations.handover import (
    HandoverBlockedError,
    HandoverService,
    IsolationWork,
    MissingAssetError,
    PressurePoint,
    PumpStatus,
    StaleHandoverError,
    ZoneEvent,
)

AT = datetime(2026, 10, 2, 8, 0, tzinfo=timezone.utc)


def _service(assets=None, events=None, points=None, pumps=None, isolation=None, controller="alice"):
    service = HandoverService()
    service.register_zone(
        "Z1",
        controller=controller,
        points=points
        if points is not None
        else [
            # corrected = 480 + 20 * 0.980665 = 499.61 kPa, inside envelope
            PressurePoint("N1", elevation_m=20.0, pressure_kpa=480.0, low_limit_kpa=450.0, high_limit_kpa=550.0),
            PressurePoint("N2", elevation_m=0.0, pressure_kpa=500.0, low_limit_kpa=450.0, high_limit_kpa=550.0),
        ],
        pumps=pumps
        if pumps is not None
        else [
            PumpStatus("P1", running=True, healthy=True, kw=42.0),
            PumpStatus("P2", running=False, healthy=True, kw=0.0),
        ],
        events=events if events is not None else [ZoneEvent("E9", "maintenance")],
        isolation=isolation
        if isolation is not None
        else [IsolationWork("WO7", frozenset({"V1", "V2"}))],
        assets=frozenset(assets) if assets is not None else None,
    )
    return service


class NormalHandoverTests(unittest.TestCase):
    def test_clean_sign_out_and_accept(self):
        service = _service()
        record = service.sign_out("Z1", "alice", at=AT)

        self.assertEqual(record.revision, 0)
        self.assertEqual(record.signed_out_by, "alice")
        self.assertIsNone(record.accepted_by)
        # handover contents
        self.assertIn("N1", record.elevation_correction_kpa)
        self.assertAlmostEqual(record.elevation_correction_kpa["N1"], 19.6133, places=4)
        self.assertAlmostEqual(record.pressure["readings"]["N1"]["corrected_kpa"], 499.6133, places=4)
        self.assertEqual(record.pressure["low"], ())
        self.assertEqual(record.pressure["high"], ())
        self.assertEqual([p.id for p in record.pumps], ["P1", "P2"])
        self.assertTrue(record.pumps[0].running)
        self.assertEqual([e.id for e in record.open_events], ["E9"])
        self.assertEqual(record.pending_isolation[0].work_order, "WO7")
        self.assertEqual(set(record.pending_isolation[0].valves), {"V1", "V2"})

        accepted = service.accept("Z1", "bob", record, at=AT)
        self.assertEqual(accepted.accepted_by, "bob")
        self.assertEqual(accepted.signed_out_by, "alice")
        self.assertEqual(service.current_view("Z1").controller, "bob")
        self.assertFalse(service.current_view("Z1").signed_out)

        actions = [(e.action, e.actor) for e in service.audit_log("Z1")]
        self.assertEqual(actions, [("sign_out", "alice"), ("accept", "bob")])
        for entry in service.audit_log("Z1"):
            self.assertEqual(entry.revision, 0)

    def test_same_operator_cannot_accept_own_handover(self):
        service = _service()
        record = service.sign_out("Z1", "alice", at=AT)
        with self.assertRaises(ValueError):
            service.accept("Z1", "alice", record)

    def test_closed_events_and_completed_isolation_excluded(self):
        service = _service(
            events=[ZoneEvent("E1", "maintenance", closed=True)],
            isolation=[IsolationWork("WO1", frozenset({"V1"}), completed=True)],
        )
        record = service.sign_out("Z1", "alice", at=AT)
        self.assertEqual(record.open_events, ())
        self.assertEqual(record.pending_isolation, ())


class StaleHandoverTests(unittest.TestCase):
    def test_concurrent_change_makes_record_stale_and_names_low_pressure(self):
        # N2 is pushed below its low limit after sign out: 420 kPa at 0 m
        service = _service()
        record = service.sign_out("Z1", "alice", at=AT)

        service.set_pressure("Z1", "N2", 420.0)  # concurrent modification bumps revision
        self.assertEqual(service.current_view("Z1").revision, 1)

        with self.assertRaises(StaleHandoverError) as caught:
            service.accept("Z1", "bob", record, at=AT)
        error = caught.exception
        self.assertEqual(error.expected_revision, 0)
        self.assertEqual(error.actual_revision, 1)
        self.assertIn("stale handover rejected", str(error))
        self.assertIn("N2", str(error))
        self.assertIn("low", str(error))
        self.assertIn("420.00", str(error))
        self.assertEqual(error.low_pressure[0].split()[0], "N2")
        self.assertEqual(error.high_pressure, ())
        # refusal clears the pending record so a fresh handover is possible
        self.assertFalse(service.current_view("Z1").signed_out)
        actions = [e.action for e in service.audit_log("Z1")]
        self.assertEqual(actions, ["sign_out", "reject"])
        self.assertIn("unresolved low pressure", service.audit_log("Z1")[-1].detail)

        new_record = service.sign_out("Z1", "alice", at=AT)
        accepted = service.accept("Z1", "bob", new_record, at=AT)
        self.assertEqual(accepted.revision, 1)

    def test_stale_record_names_high_pressure(self):
        service = _service()
        record = service.sign_out("Z1", "alice", at=AT)
        service.set_pressure("Z1", "N2", 600.0)  # above high limit
        with self.assertRaises(StaleHandoverError) as caught:
            service.accept("Z1", "bob", record, at=AT)
        self.assertIn("unresolved high pressure", str(caught.exception))
        self.assertEqual(caught.exception.high_pressure[0].split()[0], "N2")

    def test_stale_without_pressure_conditions_still_rejected(self):
        service = _service()
        record = service.sign_out("Z1", "alice", at=AT)
        service.set_pump("Z1", "P2", running=True)  # benign change, no pressure issue
        with self.assertRaises(StaleHandoverError) as caught:
            service.accept("Z1", "bob", record, at=AT)
        self.assertIn("no low/high pressure conditions", str(caught.exception))

    def test_explicit_reject_appends_audit_and_clears_pending(self):
        service = _service()
        record = service.sign_out("Z1", "alice", at=AT)
        service.reject("Z1", "bob", record, at=AT)
        self.assertFalse(service.current_view("Z1").signed_out)
        self.assertEqual([e.action for e in service.audit_log("Z1")], ["sign_out", "reject"])
        with self.assertRaises(Exception):
            service.accept("Z1", "bob", record)


class SecurityBlockTests(unittest.TestCase):
    def test_sign_out_blocked_while_security_event_active(self):
        service = _service(events=[ZoneEvent("S1", "security")])
        with self.assertRaises(HandoverBlockedError) as caught:
            service.sign_out("Z1", "alice", at=AT)
        self.assertIn("security event S1 is still active", str(caught.exception))
        self.assertFalse(service.current_view("Z1").signed_out)
        self.assertEqual([e.action for e in service.audit_log("Z1")], ["reject"])

    def test_accept_blocked_when_security_event_opens_after_sign_out(self):
        service = _service()
        record = service.sign_out("Z1", "alice", at=AT)
        service.open_event("Z1", ZoneEvent("S1", "security"))
        with self.assertRaises(HandoverBlockedError):
            service.accept("Z1", "bob", record, at=AT)
        # security blocks before the stale/version branch is even evaluated
        self.assertEqual([e.action for e in service.audit_log("Z1")], ["sign_out", "reject"])
        self.assertFalse(service.current_view("Z1").signed_out)

    def test_handover_allowed_after_security_event_closed(self):
        service = _service(events=[ZoneEvent("S1", "security", closed=True)])
        record = service.sign_out("Z1", "alice", at=AT)
        accepted = service.accept("Z1", "bob", record, at=AT)
        self.assertEqual(accepted.accepted_by, "bob")


class AssetValidationTests(unittest.TestCase):
    def test_missing_assets_block_sign_out_and_appear_in_view(self):
        service = _service(assets={"N1", "N2", "P1", "P2", "V1"})  # V2 missing
        view = service.current_view("Z1")
        self.assertEqual(view.missing_assets, ("V2",))
        with self.assertRaises(MissingAssetError) as caught:
            service.sign_out("Z1", "alice", at=AT)
        self.assertIn("V2", str(caught.exception))
        # failed validation creates no handover and no sign-out audit
        self.assertFalse(service.current_view("Z1").signed_out)
        self.assertEqual(service.audit_log("Z1"), ())

    def test_unknown_zone_rejected(self):
        with self.assertRaises(KeyError):
            HandoverService().current_view("nope")


class ViewTests(unittest.TestCase):
    def test_console_view_is_readable(self):
        service = _service()
        view = service.current_view("Z1")
        text = view.render()
        self.assertIn("PRESSURE ZONE Z1", text)
        self.assertIn("revision=0", text)
        self.assertIn("controller: alice", text)
        self.assertIn("[OK  ]", text)
        self.assertIn("[RUN] P1", text)
        self.assertIn("[STOP] P2", text)
        self.assertIn("[OPEN] E9", text)
        self.assertIn("WO7", text)
        self.assertIn("499.61 kPa", text)
        # lines tuple and render() agree
        self.assertEqual(text, "\n".join(view.lines))

    def test_view_flags_pressure_fault_and_missing_assets(self):
        service = _service(assets={"N1", "N2", "P1", "P2", "V1", "V2"})
        service.set_pressure("Z1", "N2", 420.0)
        text = service.current_view("Z1").render()
        self.assertIn("[LOW ]", text)
        service.register_zone(
            "Z2", "carol",
            points=[PressurePoint("X1", 0.0, 700.0, 450.0, 550.0)],
            assets=frozenset(),
        )
        text = service.current_view("Z2").render()
        self.assertIn("[HIGH]", text)
        self.assertIn("MISSING ASSETS: X1", text)


if __name__ == "__main__":
    unittest.main()
