#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Read-only public Vast search; prices include the requested allocated disk.

Offline inputs are one or more {queried_utc, request, response: {offers: [...]}}
snapshots, with request.allocated_storage matching --storage-gb. No credentials,
GPU libraries, SSH, instance creation, or billing endpoints are used.
"""

import argparse
import copy
import json
import math
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import Request, urlopen

API = "https://console.vast.ai/api/v0/bundles/"
GPUS = ("RTX 3090", "RTX 4090", "RTX 5090")
FIELDS = (
    "id",
    "gpu_name",
    "num_gpus",
    "gpu_frac",
    "gpu_ram",
    "cpu_arch",
    "cpu_name",
    "cpu_ram",
    "cpu_cores_effective",
    "driver_version",
    "reliability2",
    "gpu_max_power",
    "pcie_bw",
    "duration",
    "geolocation",
    "vms_enabled",
    "is_vm_deverified",
    "dph_base",
    "storage_cost",
    "storage_total_cost",
    "dph_total",
    "inet_down_cost",
    "inet_up_cost",
    "internet_down_cost_per_tb",
    "internet_up_cost_per_tb",
    "disk_space",
    "disk_bw",
    "inet_down",
    "inet_up",
)


def utc_now():
    return datetime.now(timezone.utc)


def driver_tuple(value):
    if not isinstance(value, str) or not re.fullmatch(r"\d+\.\d+(?:\.\d+)?", value):
        raise ValueError("Malformed driver version")
    return tuple(int(part) for part in value.split(".")) + (0,) * (3 - len(value.split(".")))


def query(gpu, storage_gb=200, vm=False):
    result = {
        "limit": 1000,
        "type": "on-demand",
        "gpu_name": {"eq": gpu},
        "verified": {"eq": True},
        "rentable": {"eq": True},
        "rented": {"eq": False},
        "num_gpus": {"eq": 1},
        "cpu_ram": {"gte": 64000},
        "cpu_cores_effective": {"gte": 8},
        "disk_space": {"gte": storage_gb},
        "reliability2": {"gte": 0.99},
        "driver_version": {"gte": "595.58.03"},
        "allocated_storage": storage_gb,
        "order": [["dph_total", "asc"]],
    }
    if vm:
        result["vms_enabled"] = {"eq": True}
    return result


def refresh(storage_gb=200, vm=False):
    snapshots = []
    for gpu in GPUS:
        body = query(gpu, storage_gb, vm)
        request = Request(
            API,
            json.dumps(body).encode(),
            {"Content-Type": "application/json", "User-Agent": "cloud-offers/1"},
            method="POST",
        )
        with urlopen(request, timeout=30) as response:
            payload = json.load(response)
        if not isinstance(payload.get("offers"), list):
            raise ValueError("Public search did not return an offers list")
        snapshots.append({"queried_utc": utc_now().isoformat(), "request": body, "response": payload})
    return snapshots


def qualifies(offer, gpu, storage_gb, vm, region):
    try:
        prices = [float(offer[name]) for name in ("dph_base", "storage_total_cost", "dph_total")]
        base, disk, total = prices
        return (
            offer["gpu_name"] == gpu
            and offer["num_gpus"] == 1
            and offer.get("verification") == "verified"
            and offer.get("rentable") is True
            and offer.get("rented") is False
            and float(offer["cpu_ram"]) >= 64000
            and float(offer["cpu_cores_effective"]) >= 8
            and float(offer["disk_space"]) >= storage_gb
            and float(offer["reliability2"]) >= 0.99
            and driver_tuple(offer["driver_version"]) >= (595, 58, 3)
            and offer.get("cpu_arch") in (None, "amd64", "x86_64", "x64", "x86")
            and (not vm or (offer.get("vms_enabled") is True and offer.get("is_vm_deverified") is not True))
            and (not region or offer.get("geolocation", "").split(",")[-1].strip() == region)
            and all(math.isfinite(price) for price in prices)
            and base > 0
            and disk >= 0
            and total > 0
            and math.isclose(total, base + disk, rel_tol=1e-6, abs_tol=1e-7)
            and isinstance(offer["id"], int)
            and offer["id"] > 0
        )
    except (KeyError, TypeError, ValueError, OverflowError):
        return False


def make_report(snapshots, storage_gb=200, vm=False, region=None, now=None):
    now = now or utc_now()
    if storage_gb < 200:
        raise ValueError("storage must be at least 200 GB")
    if region and not re.fullmatch(r"[A-Z]{2}", region):
        raise ValueError("region must be an exact two-letter country code, e.g. US")
    offers, queries = {gpu: {} for gpu in GPUS}, []
    for snapshot in snapshots:
        try:
            stamp = datetime.fromisoformat(snapshot["queried_utc"].replace("Z", "+00:00"))
            if stamp.tzinfo is None or not -300 <= (now - stamp).total_seconds() <= 86400:
                raise ValueError("stale or invalid snapshot timestamp (maximum age: 24 hours)")
            request = snapshot["request"]
            if request.get("allocated_storage") != storage_gb:
                raise ValueError("snapshot storage quote does not match requested storage")
            gpu = request["gpu_name"]["eq"]
            if gpu not in GPUS or not isinstance(snapshot["response"]["offers"], list):
                raise ValueError("unsupported GPU or malformed offers list")
        except (KeyError, TypeError) as error:
            raise ValueError("malformed timestamped search snapshot") from error
        index = len(queries)
        queries.append({"queried_utc": snapshot["queried_utc"], "request": copy.deepcopy(request)})
        for offer in snapshot["response"]["offers"]:
            if qualifies(offer, gpu, storage_gb, vm, region):
                row = {field: copy.deepcopy(offer.get(field)) for field in FIELDS}
                row.update(allocated_storage_gb=storage_gb, query_index=index, quoted_utc=snapshot["queried_utc"])
                offers[gpu][offer["id"]] = row
    if not queries:
        raise ValueError("at least one timestamped search snapshot is required")
    return {
        "schema_version": 1,
        "generated_utc": now.isoformat(),
        "endpoint": API,
        "queried_utc": max(query["queried_utc"] for query in queries),
        "filters": {
            "storage_gb": storage_gb,
            "vm": vm,
            "region": region,
            "max_age_hours": 24,
            "min_ram_mb": 64000,
            "min_effective_cpu": 8,
            "min_reliability": 0.99,
            "min_driver": "595.58.03",
        },
        "price_scope": "USD/hour: GPU+CPU+RAM+requested disk; excludes traffic, tax and stopped storage",
        "queries": queries,
        "offers": {
            gpu: sorted(rows.values(), key=lambda row: (row["dph_total"], row["id"]))[:3]
            for gpu, rows in offers.items()
        },
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="JSON report destination")
    parser.add_argument("--input", type=Path, nargs="+", help="Replay timestamped public search snapshot(s)")
    parser.add_argument("--storage-gb", type=int, default=200)
    parser.add_argument("--vm", action="store_true")
    parser.add_argument("--region", help="Exact uppercase country code; filtered locally, e.g. US")
    args = parser.parse_args(argv)
    try:
        if args.storage_gb < 200:
            raise ValueError("storage must be at least 200 GB")
        if args.region and not re.fullmatch(r"[A-Z]{2}", args.region):
            raise ValueError("region must be an exact two-letter country code, e.g. US")
        snapshots = []
        if args.input:
            for path in args.input:
                data = json.loads(path.read_text(encoding="utf-8-sig"))
                snapshots.extend(data if isinstance(data, list) else [data])
        else:
            snapshots = refresh(args.storage_gb, args.vm)
        report = make_report(snapshots, args.storage_gb, args.vm, args.region)
        report["source_mode"] = "offline" if args.input else "live_public_search"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    except (OSError, ValueError) as error:
        parser.exit(1, f"Offer search failed: {error}\n")


if __name__ == "__main__":
    main()
