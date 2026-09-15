"""GPU-only regression using the actual extension and a stock simulated Panda."""
from dataclasses import replace
import time
import uuid
from unittest.mock import patch
import numpy as np
from pxr import UsdGeom
from audit_cases import frames
from simdroid_data_collection.domain import Pose, sample_waypoints


async def until_idle(owner, planning_only=False):
    deadline = time.monotonic() + 60.
    while owner.task is not None or (not planning_only and owner.runner.active):
        if time.monotonic() > deadline:
            raise AssertionError(f"Randomized test exceeded its wall-time budget: {owner.message}")
        await frames(1)


async def randomization_motion_audit(owner, runtime):
    originals = tuple(owner.store.read(path) for path in owner.store.paths())
    assert len(originals) == 2
    initial = owner.robot.tcp_pose()
    delta = (0., .025, 0.) if UsdGeom.GetStageUpAxis(owner.store.stage) == "Y" else (0., 0., .025)
    targets = (Pose(np.asarray(initial.position)+delta, initial.orientation), initial)
    try:
        for wp, target, action, radius in zip(originals, targets, ("close", "open"), (.01, .007)):
            owner.store.update(replace(wp, pose=target, gripper=action, dwell=.2,
                                       randomize_position=True, randomization_radius=radius))
        await frames(20)
        nominal = owner.store.snapshot()
        authored = owner.store.layer.ExportToString()
        assert owner.robot.ready(), "Randomization authoring invalidated the physics view"
        with patch("simdroid_data_collection.extension.secrets.randbits", return_value=12345):
            owner.launch("validate")
        await until_idle(owner, planning_only=True)
        assert owner.message.startswith("Validated"), owner.message
        assert owner.validated_randomization is not None
        sampled = owner.validated_randomization[1]
        assert sampled == sample_waypoints(nominal, 12345)
        for original, actual in zip(nominal, sampled):
            distance, angle = original.pose.error(actual.pose)
            assert 0 < distance <= original.randomization_radius and angle < 1e-7
        # A run after validation must not draw a second random target set.
        with patch("simdroid_data_collection.extension.secrets.randbits",
                   side_effect=AssertionError("Run resampled successfully validated targets")):
            owner.launch("sequence")
        await until_idle(owner)
        assert owner.runner.state == "complete", owner.message
        metadata = owner.runner.metadata
        assert metadata["randomization"]["seed"] == 12345
        assert metadata["randomization"]["randomized_waypoints"] == 2
        assert [e["action"] for e in owner.runner.events if e["type"] == "gripper_action"] == ["close", "open"]
        by_path = {w.path: w for w in sampled}
        for sample in owner.runner.samples:
            goal = by_path[sample["waypoint"]]
            np.testing.assert_allclose(sample["goal_tcp"]["position"], goal.pose.position, atol=1e-12)
        assert owner.robot.tcp_pose().error(sampled[-1].pose)[0] < sampled[-1].position_tolerance
        assert owner.store.layer.ExportToString() == authored, "Execution changed nominal authoring"
        filename = str(runtime / f"randomized_run_{uuid.uuid4().hex[:8]}.json")
        owner.runner.export(filename)
        print(f"PASS randomization: native Lula Validate/Run, sampled arrival, close/open, telemetry {filename}", flush=True)

        with patch("simdroid_data_collection.extension.secrets.randbits", return_value=67890):
            owner.launch("validate")
        await until_idle(owner, planning_only=True)
        assert owner.message.startswith("Validated"), owner.message
        assert owner.validated_randomization[1] != sampled, "A subsequent run reused the consumed sample"
        owner.abort()
        for wp in nominal:
            owner.store.update_randomization(wp.path, False, wp.randomization_radius)
        await frames(12)
        owner.launch("validate")
        await until_idle(owner, planning_only=True)
        assert owner.message.startswith("Validated"), owner.message
        assert owner.validated_randomization[1] == owner.store.snapshot()
        assert owner.validated_randomization[2]["seed"] is None
        print("PASS randomization: fresh next sample and deterministic targets when disabled", flush=True)
    finally:
        owner.abort()
        for wp in originals:
            owner.store.update(wp)
        await frames(12)
