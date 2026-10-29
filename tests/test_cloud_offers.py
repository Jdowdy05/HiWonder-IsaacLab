# SPDX-License-Identifier: BSD-3-Clause
"""Public search and offline pricing checks; no GPU, credentials or live HTTP."""

import importlib.util
import io
import json
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

SCRIPT = Path(__file__).resolve().parents[1] / "cloud" / "offers.py"
NOW = datetime(2026, 10, 29, 6, tzinfo=timezone.utc)


def offer(**changes):
    result = {
        "id": 1,
        "gpu_name": "RTX 4090",
        "num_gpus": 1,
        "verification": "verified",
        "rentable": True,
        "rented": False,
        "cpu_arch": "amd64",
        "cpu_ram": 64000,
        "cpu_cores_effective": 8,
        "disk_space": 250,
        "driver_version": "595.58.03",
        "reliability2": 0.99,
        "dph_base": 0.4,
        "storage_total_cost": 0.1,
        "dph_total": 0.5,
        "geolocation": "Pennsylvania, US",
        "vms_enabled": False,
        "is_vm_deverified": False,
        "gpu_max_power": 450,
        "pcie_bw": 25.1,
        "duration": 36000,
        "inet_up_cost": 0.004,
        "inet_down_cost": 0.002,
    }
    result.update(changes)
    return result


class OfferTests(unittest.TestCase):
    def setUp(self):
        self.assertTrue(SCRIPT.is_file(), "Public offer tool is not implemented")
        spec = importlib.util.spec_from_file_location("cloud_offers", SCRIPT)
        self.module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.module)

    def snapshot(self, *offers, **changes):
        result = {
            "queried_utc": NOW.isoformat(),
            "request": {"gpu_name": {"eq": "RTX 4090"}, "allocated_storage": 200},
            "response": {"offers": list(offers)},
        }
        result.update(changes)
        return result

    def test_quote_selection_includes_disk_instead_of_sorting_gpu_base(self):
        rows = [offer(id=1, dph_base=0.2, storage_total_cost=0.5, dph_total=0.7), offer(id=2)]
        report = self.module.make_report([self.snapshot(*rows)], now=NOW)
        self.assertEqual([row["id"] for row in report["offers"]["RTX 4090"]], [2, 1])
        self.assertEqual(report["offers"]["RTX 4090"][0]["dph_total"], 0.5)
        self.assertEqual(report["queries"][0]["request"]["allocated_storage"], 200)

    def test_only_three_cheapest_qualifying_offers_per_gpu(self):
        rows = [offer(id=i, dph_base=0.1 * i, dph_total=0.1 * i + 0.1) for i in range(5, 0, -1)]
        result = self.module.make_report([self.snapshot(*rows)], now=NOW)
        self.assertEqual([row["id"] for row in result["offers"]["RTX 4090"]], [1, 2, 3])

    def test_wrong_gpu_resources_quota_disk_driver_and_reliability_are_excluded(self):
        cases = [
            {"gpu_name": "RTX 3090"},
            {"num_gpus": 2},
            {"cpu_ram": 63999},
            {"cpu_cores_effective": 7.99},
            {"cpu_cores": 64, "cpu_cores_effective": 4},
            {"disk_space": 199.9},
            {"driver_version": "595.58.02"},
            {"driver_version": "595.9.99"},
            {"driver_version": "malformed"},
            {"reliability2": 0.9899},
            {"cpu_arch": "arm64"},
            {"verification": "unverified"},
            {"rentable": False},
            {"rented": True},
        ]
        for changes in cases:
            with self.subTest(changes=changes):
                result = self.module.make_report([self.snapshot(offer(**changes))], now=NOW)
                self.assertEqual(result["offers"]["RTX 4090"], [])

    def test_driver_versions_are_numeric_and_missing_arch_is_allowed(self):
        rows = [offer(id=1, driver_version="595.112.07"), offer(id=2, driver_version="595.84", cpu_arch=None)]
        result = self.module.make_report([self.snapshot(*rows)], now=NOW)
        self.assertEqual(len(result["offers"]["RTX 4090"]), 2)

    def test_one_exclusive_gpu_on_multigpu_host_uses_its_effective_cpu_quota(self):
        # gpu_frac is the allocation's share of all host GPUs, not a fractional GPU.
        rows = [offer(id=1, gpu_frac=0.125), offer(id=2, gpu_frac=0.5, cpu_cores_effective=4)]
        result = self.module.make_report([self.snapshot(*rows)], now=NOW)
        self.assertEqual([row["id"] for row in result["offers"]["RTX 4090"]], [1])

    def test_vm_mode_excludes_unavailable_and_deverified_vms(self):
        rows = [offer(id=1), offer(id=2, vms_enabled=True, is_vm_deverified=True), offer(id=3, vms_enabled=True)]
        result = self.module.make_report([self.snapshot(*rows)], vm=True, now=NOW)
        self.assertEqual([row["id"] for row in result["offers"]["RTX 4090"]], [3])

    def test_stale_or_future_snapshot_is_rejected(self):
        for stamp in (NOW - timedelta(hours=25), NOW + timedelta(hours=1)):
            with self.subTest(stamp=stamp), self.assertRaisesRegex(ValueError, "timestamp|stale"):
                self.module.make_report([self.snapshot(offer(), queried_utc=stamp.isoformat())], now=NOW)

    def test_eight_gb_quote_cannot_be_relabelled_as_two_hundred_gb(self):
        snapshot = self.snapshot(offer())
        snapshot["request"]["allocated_storage"] = 8
        with self.assertRaisesRegex(ValueError, "storage"):
            self.module.make_report([snapshot], now=NOW)

    def test_missing_disk_component_or_inconsistent_total_is_excluded(self):
        for changes in ({"storage_total_cost": None}, {"dph_total": 0.4}, {"dph_total": float("nan")}):
            with self.subTest(changes=changes):
                result = self.module.make_report([self.snapshot(offer(**changes))], now=NOW)
                self.assertEqual(result["offers"]["RTX 4090"], [])

    def test_region_is_exact_country_code(self):
        rows = [offer(id=1), offer(id=2, geolocation="France, FR")]
        result = self.module.make_report([self.snapshot(*rows)], region="US", now=NOW)
        self.assertEqual([row["id"] for row in result["offers"]["RTX 4090"]], [1])
        with self.assertRaisesRegex(ValueError, "region"):
            self.module.make_report([self.snapshot(*rows)], region="United States", now=NOW)

    def test_qualification_does_not_mutate_input_and_retains_price_evidence(self):
        snapshot = self.snapshot(offer())
        before = json.dumps(snapshot)
        result = self.module.make_report([snapshot], now=NOW)
        self.assertEqual(json.dumps(snapshot), before)
        chosen = result["offers"]["RTX 4090"][0]
        self.assertEqual(chosen.get("allocated_storage_gb"), 200)
        self.assertEqual(result["filters"].get("storage_gb"), 200)
        self.assertEqual(chosen["inet_up_cost"], 0.004)
        self.assertEqual(chosen["gpu_max_power"], 450)
        self.assertEqual(result["schema_version"], 1)

    def test_public_refresh_only_posts_search_with_requested_storage(self):
        seen = []

        def http(request, timeout):
            self.assertEqual(request.full_url, "https://console.vast.ai/api/v0/bundles/")
            self.assertEqual(request.method, "POST")
            body = json.loads(request.data)
            self.assertEqual(body["allocated_storage"], 200)
            self.assertTrue(body["verified"]["eq"])
            self.assertTrue(body["rentable"]["eq"])
            self.assertFalse(body["rented"]["eq"])
            self.assertEqual(body["num_gpus"]["eq"], 1)
            self.assertTrue(body["vms_enabled"]["eq"])
            seen.append(body["gpu_name"]["eq"])
            return io.BytesIO(json.dumps({"offers": [offer(gpu_name=seen[-1], vms_enabled=True)]}).encode())

        with patch.object(self.module, "urlopen", http):
            snapshots = self.module.refresh(vm=True)
        result = self.module.make_report(snapshots, vm=True)
        self.assertEqual(seen, ["RTX 3090", "RTX 4090", "RTX 5090"])
        self.assertEqual(result["offers"]["RTX 3090"][0]["dph_total"], 0.5)

    def test_offline_cli_writes_json_without_contacting_search(self):
        with tempfile.TemporaryDirectory() as temporary:
            source = Path(temporary) / "input.json"
            target = Path(temporary) / "offers.json"
            source.write_text(json.dumps(self.snapshot(offer(), queried_utc=datetime.now(timezone.utc).isoformat())))
            with patch.object(self.module, "urlopen", side_effect=AssertionError("No live HTTP in offline mode")):
                try:
                    self.module.main(["--input", str(source), "--output", str(target), "--storage-gb", "200"])
                except SystemExit as error:
                    self.fail(f"The GB storage CLI did not accept a valid quote: {error}")
            self.assertEqual(json.loads(target.read_text())["offers"]["RTX 4090"][0]["id"], 1)


if __name__ == "__main__":
    unittest.main()
