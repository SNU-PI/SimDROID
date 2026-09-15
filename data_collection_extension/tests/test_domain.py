"""Run with any Python environment containing NumPy; no Kit required."""
import importlib.util
from pathlib import Path
import sys
import unittest
from dataclasses import replace
import numpy as np

SOURCE = Path(__file__).resolve().parents[1] / "exts/simdroid.data_collection/simdroid_data_collection/domain.py"
spec = importlib.util.spec_from_file_location("waypoint_domain", SOURCE)
d = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = d
spec.loader.exec_module(d)


class DomainTests(unittest.TestCase):
    def test_randomization_defaults_and_zero_radius(self):
        wp = d.Waypoint("/goal", d.Pose((1, 2, 3), (1, 0, 0, 0)))
        self.assertFalse(wp.randomize_position)
        self.assertEqual(wp.randomization_radius, .02)
        self.assertIs(d.sample_waypoints((wp,), 1)[0], wp)
        zero = replace(wp, randomize_position=True, randomization_radius=0.)
        self.assertIs(d.sample_waypoints((zero,), 2)[0], zero)
        disabled = replace(wp, enabled=False, randomize_position=True)
        self.assertIs(d.sample_waypoints((disabled,), 3)[0], disabled)

    def test_randomization_radius_validation(self):
        pose = d.Pose((0, 0, 0), (1, 0, 0, 0))
        for radius in (-.01, np.nan, np.inf, -np.inf):
            with self.subTest(radius=radius), self.assertRaisesRegex(ValueError, "radius"):
                d.Waypoint("/goal", pose, randomization_radius=radius)
        with self.assertRaisesRegex(ValueError, "boolean"):
            d.Waypoint("/goal", pose, randomize_position="false")

    def test_randomization_seed_and_immutable_nominal(self):
        wp = d.Waypoint("/goal", d.Pose((.3, .7, -.2), d.euler_quat((20, 40, 60))),
                        gripper="close", dwell=1., randomize_position=True, randomization_radius=.03)
        before = dict(vars(wp))
        a = d.sample_waypoints((wp,), 123)
        self.assertEqual(a, d.sample_waypoints((wp,), 123))
        self.assertNotEqual(a, d.sample_waypoints((wp,), 124))
        self.assertEqual(vars(wp), before)
        self.assertGreater(a[0].pose.error(wp.pose)[0], 0.)
        self.assertLessEqual(a[0].pose.error(wp.pose)[0], wp.randomization_radius)
        self.assertLess(a[0].pose.error(wp.pose)[1], 1e-7)
        for key in vars(wp):
            if key != "pose":
                self.assertEqual(getattr(a[0], key), getattr(wp, key))

    def test_randomization_is_uniform_in_volume(self):
        wp = d.Waypoint("/goal", d.Pose((2, -3, 4), (1, 0, 0, 0)),
                        randomize_position=True, randomization_radius=.025)
        samples = d.sample_waypoints((wp,)*6000, 42)
        offsets = (np.array([w.pose.position for w in samples])-wp.pose.position) / wp.randomization_radius
        radii = np.linalg.norm(offsets, axis=1)
        self.assertTrue(np.all(radii <= 1.+1e-12))
        self.assertAlmostEqual(float(np.mean(radii**3)), .5, delta=.02)
        np.testing.assert_allclose(np.mean(offsets, axis=0), 0., atol=.025)
        np.testing.assert_allclose(np.mean(offsets**2, axis=0), .2, atol=.015)

    def test_randomization_does_not_enlarge_arrival_tolerance(self):
        nominal = d.Waypoint("/goal", d.Pose((0, 0, 0), (1, 0, 0, 0)),
                             gripper="close", randomize_position=True, randomization_radius=.1)
        sampled, = d.sample_waypoints((nominal,), 42)
        gate = d.ArrivalGate(sampled)
        self.assertIsNone(gate.step(.2, nominal.pose.error(sampled.pose), True, True))
        self.assertEqual(gate.step(.2, (0, 0), True, True), "close")

    def test_tcp_inverse(self):
        goal = d.Pose((.2, -.7, .3), d.euler_quat((20, 80, -60)))
        offset = d.Pose((0, 0, .107), d.euler_quat((0, 0, 45)))
        hand = goal.compose(offset.inverse())
        np.testing.assert_allclose(hand.compose(offset).position, goal.position, atol=1e-12)
        self.assertLess(hand.compose(offset).error(goal)[1], 1e-7)

    def test_rotated_tcp_offset(self):
        hand = d.Pose((1, 2, 3), d.euler_quat((0, 90, 0)))
        result = hand.compose(d.Pose((0, 0, .107), (1, 0, 0, 0)))
        np.testing.assert_allclose(result.position, (1.107, 2, 3), atol=1e-12)

    def test_quaternion_double_cover(self):
        q = d.euler_quat((40, -20, 140))
        self.assertLess(d.angular_error(q, -q), 1e-7)
        self.assertLess(d.angular_error(q, d.slerp(q, -q, .5)), 1e-7)

    def test_large_finite_quaternion_is_normalized(self):
        q = d.unit_quat((1e308, 1e308, 0, 0))
        np.testing.assert_allclose(q, (2**-.5, 2**-.5, 0, 0))

    def test_slerp_shortest_path(self):
        middle = d.slerp(d.euler_quat((0, 0, 170)), d.euler_quat((0, 0, -170)), .5)
        self.assertLess(d.angular_error(middle, d.euler_quat((0, 0, 180))), 1e-7)

    def test_euler_roundtrip(self):
        q = d.euler_quat((45, -35, 130))
        self.assertLess(d.angular_error(q, d.euler_quat(d.quat_euler(q))), 1e-7)

    def test_validation(self):
        with self.assertRaises(ValueError):
            d.Pose((0, 0, 0), (0, 0, 0, 0))
        pose = d.Pose((0, 0, 0), (1, 0, 0, 0))
        for kwargs in ({"speed": 0}, {"dwell": -1}, {"timeout": .1}, {"gripper": "toggle"},
                       {"position_tolerance": float("nan")}):
            with self.assertRaises(ValueError):
                d.Waypoint("/goal", pose, **kwargs)

    def test_native_joint_limits_at_full_speed(self):
        limits = np.arange(1, 22, dtype=float).reshape(3, 7)
        actual = d.scaled_joint_limits(limits, 1.)
        np.testing.assert_array_equal(actual, limits)
        actual[0][0] = 999
        self.assertEqual(limits[0, 0], 1., "Returned limits must not alias the native model")

    def test_joint_limits_use_derivative_time_scaling(self):
        limits = np.arange(1, 22, dtype=float).reshape(3, 7)
        for speed in (.3, .7, 1., .3):
            actual = d.scaled_joint_limits(limits, speed)
            for i in range(3):
                np.testing.assert_allclose(actual[i], limits[i]*speed**(i+1))
        np.testing.assert_array_equal(limits, np.arange(1, 22).reshape(3, 7))

    def test_invalid_joint_limits_and_speed_are_rejected(self):
        for limits in (np.ones((3, 6)), np.zeros((3, 7)), -np.ones((3, 7)),
                       np.full((3, 7), np.nan), np.full((3, 7), np.inf)):
            with self.assertRaises(ValueError):
                d.scaled_joint_limits(limits, 1.)
        for speed in (0, -1, 1.01, np.nan, np.inf):
            with self.assertRaisesRegex(ValueError, "Speed"):
                d.scaled_joint_limits(np.ones((3, 7)), speed)

    def test_arrival_action_once_and_dwell(self):
        wp = d.Waypoint("/goal", d.Pose((0, 0, 0), (1, 0, 0, 0)), gripper="close", dwell=.2)
        gate = d.ArrivalGate(wp)
        self.assertIsNone(gate.step(.1, (0, 0), True, True))
        self.assertIsNone(gate.step(.1, (.1, 0), True, True))
        self.assertIsNone(gate.step(.1, (0, 0), True, True))
        self.assertEqual(gate.step(.1, (0, 0), True, True), "close")
        self.assertIsNone(gate.step(.1, (0, 0), True, False))
        self.assertFalse(gate.complete)
        self.assertIsNone(gate.step(.1, (0, 0), True, True))
        self.assertIsNone(gate.step(.1, (0, 0), True, True))
        self.assertTrue(gate.complete)
        self.assertIsNone(gate.step(.1, (0, 0), True, True))

    def test_orientation_motion_and_timeout_gates(self):
        wp = d.Waypoint("/goal", d.Pose((0, 0, 0), (1, 0, 0, 0)), timeout=1.)
        gate = d.ArrivalGate(wp)
        self.assertIsNone(gate.step(.3, (0, 1), True, True))
        self.assertIsNone(gate.step(.3, (0, 0), False, True))
        with self.assertRaises(TimeoutError):
            gate.step(.5, (0, 1), True, True)
        with self.assertRaises(ValueError):
            d.ArrivalGate(wp).step(0, (0, 0), True, True)


if __name__ == "__main__":
    unittest.main()
