"""Native USD-only regressions; no Kit app, rendering, PhysX simulation or CUDA.

Run with the installed omni.usd.libs on PYTHONPATH and its bin directory plus
the Python environment's lib directory on LD_LIBRARY_PATH. Kit's command
dispatcher is stubbed; the real extension undo command and USD methods run.
"""
from dataclasses import replace
import importlib.util
from pathlib import Path
import sys
import tempfile
import types
import unittest
import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade

SOURCE = Path(__file__).resolve().parents[1] / "exts/simdroid.data_collection/simdroid_data_collection"
omni = types.ModuleType("omni")
omni.kit = types.ModuleType("omni.kit")
omni.kit.commands = types.ModuleType("omni.kit.commands")
omni.kit.commands.Command = object
for module in (omni, omni.kit, omni.kit.commands):
    sys.modules[module.__name__] = module
package = types.ModuleType("usd_randomization_test")
package.__path__ = [str(SOURCE)]
sys.modules[package.__name__] = package
for name in ("domain", "stage_store", "preview"):
    spec = importlib.util.spec_from_file_location(f"{package.__name__}.{name}", SOURCE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
from usd_randomization_test.domain import Pose, euler_quat, sample_waypoints
from usd_randomization_test.stage_store import StageStore, FrankaWaypointLayerEdit, set_world_pose, world_pose
from usd_randomization_test.preview import PreviewManager


class USDTests(unittest.TestCase):
    def assert_waypoint_equal(self, actual, expected):
        distance, angle = actual.pose.error(expected.pose)
        self.assertLess(distance, 1e-12)
        self.assertLess(angle, 1e-7)
        self.assertEqual(replace(actual, pose=expected.pose), expected)

    def setUp(self):
        self.commands = []
        def execute(name, **kwargs):
            self.assertEqual(name, "FrankaWaypointLayerEdit")
            command = FrankaWaypointLayerEdit(**kwargs)
            command.do()
            self.commands.append(command)
            return True, None
        omni.kit.commands.execute = execute

    def fixture(self, units=1., up="Y"):
        stage = Usd.Stage.CreateInMemory()
        UsdGeom.SetStageMetersPerUnit(stage, units)
        UsdGeom.SetStageUpAxis(stage, up)
        UsdGeom.Xform.Define(stage, "/World")
        store = StageStore(stage)
        store.create_sequence()
        pose = Pose((.3, .8, -.2), euler_quat((15, 30, 45)))
        path = store.add(pose)
        return stage, store, path

    def test_legacy_duplicate_save_load_and_undo(self):
        stage, store, path = self.fixture()
        # Older files lack both attributes, and must keep deterministic targets.
        with Usd.EditContext(stage, store.layer):
            stage.GetPrimAtPath(path).RemoveProperty("frankaPath:randomize_position")
            stage.GetPrimAtPath(path).RemoveProperty("frankaPath:randomization_radius")
        original = store.read(path)
        self.assertFalse(original.randomize_position)
        self.assertEqual(original.randomization_radius, .02)
        wp = replace(original, randomize_position=True, randomization_radius=.037)
        store.update(wp)
        self.commands[-1].undo()
        self.assertEqual(store.read(path), original)
        self.commands[-1].do()
        self.assert_waypoint_equal(store.read(path), wp)
        clone = store.add(wp.pose, source=path)
        self.assertEqual(store.read(clone).randomization_radius, .037)
        self.assertTrue(store.read(clone).randomize_position)
        authored = store.layer.ExportToString()
        sample_waypoints(store.snapshot(), 123)
        self.assertEqual(store.layer.ExportToString(), authored)
        with tempfile.TemporaryDirectory(prefix="franka-randomization-test-") as folder:
            filename = str(Path(folder) / "waypoints.usda")
            store.save(filename)
            other = Usd.Stage.CreateInMemory()
            UsdGeom.SetStageMetersPerUnit(other, 1.)
            UsdGeom.SetStageUpAxis(other, "Y")
            loaded = StageStore(other)
            loaded.load(filename)
            self.assert_waypoint_equal(loaded.read(path), wp)

    def test_regions_units_transforms_opacity_and_lifecycle(self):
        for units, up in ((1., "Y"), (.01, "Z")):
            with self.subTest(units=units, up=up):
                stage, store, path = self.fixture(units, up)
                preview = PreviewManager(stage)
                preview.update(store)
                marker = preview.marker_paths[path]
                region = marker + "/RandomizationRegion"
                self.assertFalse(stage.GetPrimAtPath(region))
                wp = replace(store.read(path), randomize_position=True, randomization_radius=.05)
                store.update(wp)
                original_target = stage.GetEditTarget()
                preview.update(store)
                self.assertEqual(stage.GetEditTarget(), original_target)
                sphere = UsdGeom.Sphere.Get(stage, region)
                self.assertAlmostEqual(sphere.GetRadiusAttr().Get(), .05/units)
                np.testing.assert_allclose(world_pose(sphere.GetPrim()).position, wp.pose.position)
                self.assertEqual(preview.selection_target(sphere.GetPrim()), path)
                material = UsdShade.MaterialBindingAPI(sphere.GetPrim()).ComputeBoundMaterial()[0]
                shader = UsdShade.Shader.Get(stage, str(material.GetPath()) + "/Surface")
                self.assertAlmostEqual(shader.GetInput("opacity").Get(), .15, places=6)
                for prim in Usd.PrimRange(stage.GetPrimAtPath(preview.root)):
                    self.assertFalse(prim.HasAPI(UsdPhysics.RigidBodyAPI))
                    self.assertFalse(prim.HasAPI(UsdPhysics.CollisionAPI))
                self.assertNotIn(preview.root, stage.GetRootLayer().ExportToString())
                self.assertNotIn(preview.root, store.layer.ExportToString())
                with Usd.EditContext(stage, store.layer):
                    set_world_pose(stage.GetPrimAtPath(store.sequence),
                                   Pose((.1, .2, .3), euler_quat((10, 20, 30))))
                preview.update(store)
                np.testing.assert_allclose(world_pose(sphere.GetPrim()).position, store.read(path).pose.position)
                store.update(replace(store.read(path), randomization_radius=.08))
                preview.update(store)
                self.assertAlmostEqual(sphere.GetRadiusAttr().Get(), .08/units)
                preview.set_hidden(True)
                self.assertEqual(sphere.ComputeVisibility(), "invisible")
                preview.set_hidden(False)
                store.update(replace(store.read(path), randomize_position=False))
                preview.update(store)
                self.assertFalse(stage.GetPrimAtPath(region))
                store.update(replace(store.read(path), randomize_position=True, randomization_radius=0))
                preview.update(store)
                self.assertFalse(stage.GetPrimAtPath(region))
                store.update(replace(store.read(path), randomization_radius=.02))
                preview.update(store)
                store.remove(path)
                preview.update(store)
                self.assertFalse(stage.GetPrimAtPath(marker))
                preview.destroy()
                self.assertFalse(stage.GetPrimAtPath(preview.root))

    def test_stronger_layer_cannot_silently_override_randomization(self):
        stage, store, path = self.fixture()
        original = store.layer.ExportToString()
        with Usd.EditContext(stage, stage.GetRootLayer()):
            stage.GetPrimAtPath(path).GetAttribute("frankaPath:randomization_radius").Set(.01)
        with self.assertRaisesRegex(ValueError, "stronger USD layer"):
            store.update(replace(store.read(path), randomization_radius=.02))
        self.assertEqual(store.layer.ExportToString(), original)

    def test_settings_only_edit_preserves_exact_authored_transform(self):
        stage, store, path = self.fixture()
        transform = stage.GetPrimAtPath(path).GetAttribute("xformOp:transform").Get()
        for enabled, radius in ((True, .03), (False, .03), (True, 0.), (True, .005)):
            store.update_randomization(path, enabled, radius)
            self.assertEqual(stage.GetPrimAtPath(path).GetAttribute("xformOp:transform").Get(), transform)
            self.assertEqual(store.read(path).randomize_position, enabled)
            self.assertEqual(store.read(path).randomization_radius, radius)
        before = store.layer.ExportToString()
        with self.assertRaisesRegex(ValueError, "radius"):
            store.update_randomization(path, True, -1.)
        self.assertEqual(store.layer.ExportToString(), before)


if __name__ == "__main__":
    unittest.main()
