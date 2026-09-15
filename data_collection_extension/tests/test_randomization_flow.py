"""Real orchestration/UI callbacks with fake Kit services; CPU only, no renderer."""
import asyncio
from dataclasses import replace
import importlib.util
import json
from pathlib import Path
import sys
import types
import unittest
from unittest.mock import patch
import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "exts/simdroid.data_collection/simdroid_data_collection"
PACKAGE = "randomization_flow_test"
package = types.ModuleType(PACKAGE)
package.__path__ = [str(SOURCE)]
sys.modules[PACKAGE] = package


def load(name):
    spec = importlib.util.spec_from_file_location(f"{PACKAGE}.{name}", SOURCE / f"{name}.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


domain = load("domain")
runner_module = load("runner")
stubs = {}
for name in ("carb", "omni", "omni.ext", "omni.kit", "omni.kit.app", "omni.kit.commands",
             "omni.kit.menu", "omni.kit.menu.utils", "omni.physx", "omni.timeline", "omni.usd", "omni.ui", "pxr"):
    module = types.ModuleType(name)
    stubs[name] = module
    if "." in name:
        parent, child = name.rsplit(".", 1)
        setattr(stubs[parent], child, module)
stubs["omni.ext"].IExt = object
stubs["carb"].log_info = stubs["carb"].log_warn = lambda text: None
for name in ("Tf", "Usd", "UsdGeom"):
    setattr(stubs["pxr"], name, types.SimpleNamespace())
for name in ("MenuItemDescription", "add_menu_items", "remove_menu_items"):
    setattr(stubs["omni.kit.menu.utils"], name, lambda *a, **kw: None)
for name, attrs in {
    "stage_store": ("StageStore", "FrankaWaypointLayerEdit", "find_robots", "DEFAULT_TCP"),
    "preview": ("PreviewManager",), "robot": ("FrankaBinding",), "edit_router": ("WaypointEditRouter",),
}.items():
    module = types.ModuleType(f"{PACKAGE}.{name}")
    for attr in attrs:
        setattr(module, attr, None)
    stubs[module.__name__] = module
with patch.dict(sys.modules, stubs):
    ui_module = load("ui")
    extension_module = load("extension")


class Store:
    def __init__(self):
        self.waypoints = (domain.Waypoint("/goal", domain.Pose((.3, .5, .7), (1, 0, 0, 0)),
                                           randomize_position=True),)

    def snapshot(self, selected=None):
        return tuple(w for w in self.waypoints if not selected or w.path == selected)

    def read(self, path):
        return next(w for w in self.waypoints if w.path == path)

    def update(self, wp):
        self.waypoints = tuple(wp if w.path == wp.path else w for w in self.waypoints)

    def update_randomization(self, path, enabled, radius):
        self.update(replace(self.read(path), randomize_position=enabled, randomization_radius=radius))


class Robot:
    def __init__(self):
        self.path = "/robot"
        self.tcp = self.pose = domain.Pose((0, 0, 0), (1, 0, 0, 0))
        self.stage = types.SimpleNamespace(GetPrimAtPath=lambda p: types.SimpleNamespace(IsActive=lambda: True))
        self.art = types.SimpleNamespace(get_joint_positions=lambda: np.zeros(7),
                                         get_joint_velocities=lambda: np.zeros(7))
        self.gripper_target = np.zeros(2)
        self.received = []
        self.fail = False

    ready = lambda self: True
    arm_positions = lambda self: np.zeros(7)
    base_pose = lambda self: self.tcp
    tcp_pose = lambda self: self.pose
    command_arm = lambda self, q: None
    tick_gripper = lambda self, dt: None
    command_gripper = lambda self, action: None
    gripper_ready = lambda self: True

    async def plan(self, targets, progress):
        self.received.append(targets)
        await asyncio.sleep(0)
        if self.fail:
            raise ValueError("Unreachable test target")
        return [types.SimpleNamespace(waypoint=w, start=np.zeros(7), end=np.zeros(7),
                                      trajectory=None, duration=0.) for w in targets]


def make_owner():
    owner = extension_module.Extension()
    owner.store, owner.robot = Store(), Robot()
    owner.selected = "/goal"
    owner.task = owner.ui = owner.validated_randomization = None
    owner.message, owner.dirty = "", False
    owner.timeline = types.SimpleNamespace(is_playing=lambda: True)
    owner.preview = types.SimpleNamespace(set_hidden=lambda hidden: None)
    owner.reconcile_configuration = lambda: None
    owner.configuration = lambda: ("/sequence", "/robot", owner.robot.tcp)
    owner.ensure_robot = lambda physics=False: owner.robot
    owner.runner = runner_module.SequenceRunner(owner.set_status)
    return owner


class FlowTests(unittest.IsolatedAsyncioTestCase):
    async def launch(self, owner, mode):
        owner.launch(mode)
        await owner.task

    async def test_validate_reuses_sample_once_then_next_run_resamples(self):
        owner = make_owner()
        nominal = owner.store.waypoints
        with patch.object(extension_module.secrets, "randbits", side_effect=(123, 456)) as seeds:
            await self.launch(owner, "validate")
            targets = owner.robot.received[-1]
            self.assertIsNotNone(owner.validated_randomization)
            await self.launch(owner, "sequence")
            self.assertEqual(owner.robot.received[-1], targets)
            self.assertEqual(owner.runner.metadata["randomization"]["seed"], 123)
            self.assertEqual(owner.runner.metadata["waypoints"][0]["pose"], vars(targets[0].pose))
            self.assertEqual(owner.runner.metadata["randomization"]["nominal_waypoints"][0]["pose"], vars(nominal[0].pose))
            json.dumps(owner.runner.metadata, allow_nan=False)
            owner.robot.pose = targets[0].pose
            for _ in range(12):
                owner.runner.step(.1)
            self.assertEqual(owner.runner.state, "complete")
            self.assertEqual(owner.runner.segment.waypoint, targets[0])
            await self.launch(owner, "sequence")
            self.assertNotEqual(owner.robot.received[-1], targets)
            self.assertEqual(seeds.call_count, 2)
            self.assertEqual(owner.store.waypoints, nominal)
            owner.abort()

    async def test_edit_or_run_scope_invalidates_validated_sample(self):
        owner = make_owner()
        owner.store.waypoints += (replace(owner.store.waypoints[0], path="/second"),)
        with patch.object(extension_module.secrets, "randbits", side_effect=(1, 2, 3, 4)):
            await self.launch(owner, "validate")
            await self.launch(owner, "selected")
            self.assertEqual(owner.runner.metadata["randomization"]["seed"], 2)
            owner.abort()
            await self.launch(owner, "validate")
            owner.store.update(replace(owner.store.read("/goal"), randomization_radius=.04))
            await self.launch(owner, "sequence")
            self.assertEqual(owner.runner.metadata["randomization"]["seed"], 4)
            owner.abort()

    async def test_disabled_randomization_does_not_draw_seed_or_change_targets(self):
        owner = make_owner()
        owner.store.update(replace(owner.store.read("/goal"), randomize_position=False))
        with patch.object(extension_module.secrets, "randbits") as seeds:
            await self.launch(owner, "sequence")
            self.assertEqual(owner.robot.received[-1], owner.store.waypoints)
            self.assertIsNone(owner.runner.metadata["randomization"]["seed"])
            seeds.assert_not_called()
            owner.abort()

    async def test_cancel_or_failure_preserves_previous_recording(self):
        owner = make_owner()
        await self.launch(owner, "sequence")
        owner.abort()
        previous = owner.runner.metadata
        owner.launch("validate")
        task = owner.task
        owner.abort()
        await asyncio.gather(task, return_exceptions=True)
        self.assertIs(owner.runner.metadata, previous)
        self.assertIsNone(owner.validated_randomization)
        owner.robot.fail = True
        await self.launch(owner, "validate")
        self.assertIn("Planning failed", owner.message)
        self.assertIsNone(owner.validated_randomization)
        self.assertIs(owner.runner.metadata, previous)


class Model:
    def __init__(self, value):
        self.value = value

    as_bool = property(lambda self: bool(self.value))
    as_float = property(lambda self: float(self.value))

    def set_value(self, value):
        self.value = value


class UICallbackTests(unittest.TestCase):
    def setUp(self):
        self.owner = make_owner()
        self.ui = ui_module.EditorUI.__new__(ui_module.EditorUI)
        self.ui.owner = self.owner
        self.ui._saving_randomization = False
        self.ui.randomize, self.ui.random_radius = Model(True), Model(.04)
        self.ui.random_radius_field = types.SimpleNamespace(enabled=True)
        self.ui.snapshot = lambda: self.owner.store.waypoints

    def commit(self):
        self.ui.commit_randomization("/goal", self.ui.randomize, self.ui.random_radius)

    def test_immediate_randomization_edit_preserves_pose_and_other_settings(self):
        before = self.owner.store.read("/goal")
        self.commit()
        self.assertEqual(self.owner.store.read("/goal"), replace(before, randomization_radius=.04))
        self.assertEqual(self.ui._signature, self.owner.store.waypoints)
        self.ui.randomize.set_value(False)
        self.commit()
        self.assertFalse(self.owner.store.read("/goal").randomize_position)
        self.assertFalse(self.ui.random_radius_field.enabled)

    def test_invalid_radius_restores_widgets_and_keeps_authored_values(self):
        before = self.owner.store.waypoints
        self.ui.random_radius.set_value(-1)
        self.commit()
        self.assertEqual(self.owner.store.waypoints, before)
        self.assertEqual(self.ui.random_radius.as_float, .02)
        self.assertTrue(self.owner.message.startswith("ERROR"))

    def test_busy_and_stale_widgets_cannot_modify_waypoints(self):
        before = self.owner.store.waypoints
        self.owner.runner.state = "running"
        self.commit()
        self.assertEqual(self.owner.store.waypoints, before)
        self.owner.runner.state = "idle"
        self.ui.commit_randomization("/goal", Model(True), Model(.5))
        self.assertEqual(self.owner.store.waypoints, before)
        self.owner.selected = "/another"
        self.commit()
        self.assertEqual(self.owner.store.waypoints, before)


if __name__ == "__main__":
    unittest.main()
