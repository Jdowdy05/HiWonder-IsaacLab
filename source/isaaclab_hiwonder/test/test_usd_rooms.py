# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""USD ownership, support geometry, and lightweight clutter contracts."""

import unittest

from isaaclab_hiwonder.sim.rooms import MaterialSample, RoomParameters, author_room, update_material

from pxr import Usd, UsdGeom, UsdPhysics, UsdShade


class TestRoomAuthoring(unittest.TestCase):
    """Exercise authored scene properties independently of a renderer implementation."""

    def test_room_wall_clearance_and_actual_support_surface(self):
        stage = Usd.Stage.CreateInMemory()
        paths = author_room(stage, "/World/envs/env_0/Room", RoomParameters(table_shape="ellipse"))
        cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
        table_range = cache.ComputeWorldBound(stage.GetPrimAtPath(paths.table)).ComputeAlignedBox()
        self.assertAlmostEqual(table_range.GetMax()[2], 0.75, places=6)
        self.assertAlmostEqual(table_range.GetMin()[2], 0.71, places=6)
        self.assertLess(table_range.GetMax()[0], 3.6576 / 2 - 0.3)
        self.assertTrue(stage.GetPrimAtPath(paths.table).HasAPI(UsdPhysics.CollisionAPI))
        self.assertEqual(len(paths.walls), 4)
        wall = cache.ComputeWorldBound(stage.GetPrimAtPath(paths.walls[0])).ComputeAlignedBox()
        self.assertAlmostEqual(wall.GetMin()[0], -3.6576 / 2 - 0.05, places=5)
        self.assertAlmostEqual(wall.GetMax()[0], -3.6576 / 2, places=5)
        self.assertAlmostEqual(wall.GetMax()[2], 3.6576, places=5)

    def test_room_material_and_light_ownership_survives_one_room_update(self):
        stage = Usd.Stage.CreateInMemory()
        first = author_room(stage, "/World/envs/env_0/Room", RoomParameters())
        second = author_room(stage, "/World/envs/env_1/Room", RoomParameters())
        other_shader = UsdShade.Shader(stage.GetPrimAtPath(second.materials["table"] + "/Shader"))
        original = other_shader.GetInput("diffuse_color_constant").Get()
        update_material(stage, first.materials["table"], MaterialSample(color=(0.95, 0.01, 0.02)))
        self.assertEqual(other_shader.GetInput("diffuse_color_constant").Get(), original)
        table = stage.GetPrimAtPath(first.table)
        bound, _ = UsdShade.MaterialBindingAPI(table).ComputeBoundMaterial()
        self.assertEqual(str(bound.GetPath()), first.materials["table"])
        light = stage.GetPrimAtPath(first.lights[0])
        self.assertEqual(light.GetAttribute("collection:lightLink:includeRoot").Get(), False)
        self.assertEqual(light.GetRelationship("collection:lightLink:includes").GetTargets(), ["/World/envs/env_0"])

    def test_confetti_is_instanced_above_table_without_piece_rigid_bodies(self):
        stage = Usd.Stage.CreateInMemory()
        paths = author_room(stage, "/World/envs/env_0/Room", RoomParameters(confetti_count=37, seed=9))
        instancer = UsdGeom.PointInstancer(stage.GetPrimAtPath(paths.confetti))
        positions = instancer.GetPositionsAttr().Get()
        self.assertEqual(len(positions), 37)
        self.assertTrue(all(0.7501 < p[2] < 0.753 for p in positions))
        self.assertTrue(all(abs(p[0]) < 0.5 and abs(p[1]) < 0.4 for p in positions))
        self.assertFalse(any(p.HasAPI(UsdPhysics.RigidBodyAPI) for p in Usd.PrimRange(instancer.GetPrim())))


if __name__ == "__main__":
    unittest.main()
