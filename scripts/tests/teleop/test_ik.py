import unittest

import numpy as np

from quest_vr.kinematics import nova2_fk, rotation_matrix, transform_pose6
from quest_vr.local_ik import LocalIKSolver


class LocalIKTests(unittest.TestCase):
    def setUp(self):
        self.reference = np.deg2rad([88.7644, 2.15, 86.9032, -0.6098, -88.136, 4.7442])
        pose = transform_pose6(nova2_fk(self.reference))
        self.solver = LocalIKSolver.from_reference(pose, self.reference)

    def test_reference_pose_needs_no_iteration(self):
        pose = transform_pose6(self.solver.forward(self.reference))
        result = self.solver.solve(pose, self.reference)
        self.assertTrue(result.success)
        self.assertEqual(result.iterations, 0)

    def test_nearby_pose_converges(self):
        expected = self.reference + np.deg2rad([2, -1, 1.5, 0.5, -1, 2])
        pose = transform_pose6(self.solver.forward(expected))
        result = self.solver.solve(pose, self.reference)
        self.assertTrue(result.success, result)
        self.assertLess(result.position_error_mm, 0.5)
        self.assertLess(result.orientation_error_deg, 0.2)

    def test_three_orientation_axes_converge_without_translation(self):
        current = self.solver.forward(self.reference)
        for axis in np.eye(3):
            with self.subTest(axis=axis.tolist()):
                target = current.copy()
                target[:3, :3] = current[:3, :3] @ rotation_matrix(
                    axis * np.deg2rad(3.0)
                )
                result = self.solver.solve(
                    transform_pose6(target), self.reference
                )
                self.assertTrue(result.success, result)
                self.assertLess(result.position_error_mm, 0.5)
                self.assertLess(result.orientation_error_deg, 0.2)


if __name__ == "__main__":
    unittest.main()
