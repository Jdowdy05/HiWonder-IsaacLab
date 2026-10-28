# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Real loopback boundary tests for inspection ownership and local control safety."""

import io
import json
import time
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import numpy as np
import pytest
from isaaclab_hiwonder.inspection import InspectionServer
from PIL import Image


def _post(server, body, *, token=True, origin=None):
    headers = {"Content-Type": "application/json"}
    if token:
        headers["X-Inspection-Token"] = server.token
    if origin:
        headers["Origin"] = origin
    request = Request(server.url + "api/control", data=body, headers=headers, method="POST")
    try:
        with urlopen(request, timeout=2) as response:
            return response.status, json.load(response)
    except HTTPError as error:
        return error.code, json.load(error)


def _wait_for_export(server, minimum_sequence):
    deadline = time.monotonic() + 5
    while True:
        with urlopen(server.url + "api/status", timeout=2) as response:
            status = json.load(response)
        if status["export"]["sequence"] >= minimum_sequence:
            return status
        assert time.monotonic() < deadline, status
        time.sleep(0.01)


def test_http_controls_are_authenticated_validated_and_only_queued(tmp_path):
    """A browser cannot bypass bounds or execute simulator work on an HTTP thread."""
    with InspectionServer(tmp_path, num_envs=2, port=0) as server:
        command = b'{"action":"lighting","controls":{"saturation":0.4,"radius_m":0.05}}'
        assert _post(server, command, token=False)[0] == 403
        assert _post(server, command, origin="https://untrusted.example")[0] == 403
        assert server.drain_commands() == []
        assert _post(server, command, origin=server.url.rstrip("/"))[0] == 202
        assert server.drain_commands() == [{"action": "lighting", "controls": {"saturation": 0.4, "radius_m": 0.05}}]
        for body in (
            b'{"action":"lighting","controls":{"saturation":1.01}}',
            b'{"action":"lighting","controls":{"power":NaN}}',
            b'{"action":"lighting","controls":{"speed":true}}',
            b'{"action":"lighting","controls":{"shell":"echo bad"}}',
            b'{"action":"room","index":2}',
            b'{"action":"pause","paused":1}',
            b'{"action":"reset","path":"anywhere"}',
            b'{"action":"lighting","controls":{"power":' + b"9" * 512 + b"}}",
            b"[" * 1100 + b"]" * 1100,
            b"[]",
            b'{"action":',
        ):
            assert _post(server, body)[0] == 400, body
        assert _post(server, b" " * 4097)[0] == 413
        assert server.drain_commands() == []
        for body in (
            b'{"action":"room","index":1}',
            b'{"action":"pause","paused":true}',
            b'{"action":"reset"}',
            b'{"action":"clear"}',
            b'{"action":"stop"}',
        ):
            assert _post(server, body)[0] == 202
        assert [command["action"] for command in server.drain_commands()] == ["room", "pause", "reset", "clear", "stop"]
        for _ in range(256):
            code, _ = _post(server, b'{"action":"reset"}')
            if code == 429:
                break
            assert code == 202
        else:
            pytest.fail("Inspection control queue is unbounded")
        assert 0 < len(server.drain_commands()) < 256


def test_frame_publication_owns_pixels_and_preserves_full_resolution_losslessly(tmp_path):
    """Pinned full-resolution PNG pairs survive reused buffers and newer exports."""
    raw = np.zeros((480, 640, 3), dtype=np.uint8)
    raw[3, 7] = (13, 149, 251)
    policy = np.full_like(raw, 96)
    with InspectionServer(tmp_path, num_envs=1, port=0) as server:
        server.publish_status({"room": 0, "mode": "zero_action_hold"})
        server.submit_frames(raw, policy)
        raw.fill(0)
        policy.fill(0)
        status = _wait_for_export(server, 1)
        with urlopen(server.url + "wrist_raw.png?sequence=1", timeout=2) as response:
            image = Image.open(io.BytesIO(response.read()))
            assert image.size == (640, 480)
            assert image.getpixel((7, 3)) == (13, 149, 251)
        assert Image.open(tmp_path / "wrist_policy.png").getpixel((7, 3)) == (96, 96, 96)
        assert status["mode"] == "zero_action_hold"
        # A second export between paired GETs must not silently change the first pair.
        server.submit_frames(raw, policy)
        _wait_for_export(server, 2)
        with urlopen(server.url + "wrist_policy.png?sequence=1", timeout=2) as response:
            assert Image.open(io.BytesIO(response.read())).getpixel((7, 3)) == (96, 96, 96)
        with urlopen(server.url + "wrist_raw.png?sequence=1", timeout=2) as response:
            assert Image.open(io.BytesIO(response.read())).getpixel((7, 3)) == (13, 149, 251)
        with urlopen(server.url + "wrist_policy.png?sequence=2", timeout=2) as response:
            assert Image.open(io.BytesIO(response.read())).getpixel((7, 3)) == (0, 0, 0)
        # History is bounded; an expired pin must fail instead of returning different pixels.
        for sequence in (3, 4):
            server.submit_frames(raw, policy)
            _wait_for_export(server, sequence)
        with pytest.raises(HTTPError) as stale:
            urlopen(server.url + "wrist_policy.png?sequence=1", timeout=2)
        assert stale.value.code == 410
        for query in ("sequence=-1", "sequence=bad", "sequence=3&sequence=4"):
            with pytest.raises(HTTPError) as malformed:
                urlopen(server.url + "wrist_raw.png?" + query, timeout=2)
            assert malformed.value.code == 400
        with pytest.raises(HTTPError) as error:
            urlopen(server.url + "../../private.txt", timeout=2)
        assert error.value.code == 404
        server.publish_status({"room": 0, "mode": "stopped"})
    assert json.loads((tmp_path / "live_status.json").read_text())["mode"] == "stopped"
    with pytest.raises(OSError):
        urlopen(server.url + "api/status", timeout=0.2)
