# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Protect compound-filter disambiguation and non-cancelling actor evidence."""

import numpy as np
import pytest
from isaaclab_hiwonder.sim.contact_filtering import actor_contact_presence, contact_filter_layout


def test_ambiguous_actor_names_are_rejected_and_compound_contacts_do_not_cancel():
    root = "{ENV_REGEX_NS}/Room/Task"
    actors = {"shape_0": root + "/shape_0", "socket": root + "/socket", "table": "/Table"}
    colliders = {
        "shape_0": [root + "/shape_0/left", root + "/shape_0/right"],
        "socket": [root + "/socket/shape_0", root + "/socket/floor"],
        "table": ["/Table"],
    }
    with pytest.raises(ValueError, match="aliases a collider leaf name"):
        contact_filter_layout(actors, colliders)
    colliders["socket"][0] = root + "/socket/collision_socket_0"
    layout = contact_filter_layout(actors, colliders)
    assert layout["shape_0"] == [actors["shape_0"]]
    assert layout["socket"] == [actors["socket"]]
    assert layout["table"] == ["/Table"]
    forces = np.zeros((2, 1, 3, 3))
    forces[0, 0, 0, 0] = 2
    forces[0, 0, 1, 0] = -2
    forces[1, 0, 2, 2] = 0.2943
    contacts = actor_contact_presence(forces, ["shape_0", "shape_0", "table"])
    np.testing.assert_array_equal(contacts["shape_0"], [True, False])
    np.testing.assert_array_equal(contacts["table"], [False, True])
    with pytest.raises(ValueError, match="mapping mismatch"):
        actor_contact_presence(forces, ["shape_0", "table"])
