"""Fast local damped-least-squares inverse kinematics for the Nova 2."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from quest_vr.kinematics import nova2_fk, pose6_transform, rotation_vector


@dataclass(frozen=True)
class LocalIKResult:
    joints: np.ndarray
    success: bool
    iterations: int
    position_error_mm: float
    orientation_error_deg: float
    reason: str


class LocalIKSolver:
    """Solve follower IK locally while staying on the previous joint branch."""

    def __init__(
        self,
        base_transform: np.ndarray,
        *,
        damping: float = 0.03,
        max_iterations: int = 12,
        position_tolerance_mm: float = 0.5,
        orientation_tolerance_deg: float = 0.2,
        max_iteration_step_deg: float = 5.0,
        difference_step_rad: float = 1e-4,
    ) -> None:
        self.base_transform = np.asarray(base_transform, dtype=float)
        self.damping = float(damping)
        self.max_iterations = int(max_iterations)
        self.position_tolerance_mm = float(position_tolerance_mm)
        self.orientation_tolerance_rad = float(np.deg2rad(orientation_tolerance_deg))
        self.max_iteration_step_rad = float(np.deg2rad(max_iteration_step_deg))
        self.difference_step_rad = float(difference_step_rad)

    @classmethod
    def from_reference(cls, reference_pose: np.ndarray, reference_joints: np.ndarray) -> LocalIKSolver:
        """Align the DH model to one controller-confirmed pose and joint solution."""
        actual = pose6_transform(np.asarray(reference_pose, dtype=float))
        model = nova2_fk(np.asarray(reference_joints, dtype=float))
        return cls(actual @ np.linalg.inv(model))

    def forward(self, joints: np.ndarray) -> np.ndarray:
        return self.base_transform @ nova2_fk(np.asarray(joints, dtype=float))

    @staticmethod
    def _error(target: np.ndarray, current: np.ndarray) -> tuple[np.ndarray, float, float]:
        position_mm = target[:3, 3] - current[:3, 3]
        # Spatial/world-frame rotation error, matching the numerical Jacobian.
        orientation = rotation_vector(target[:3, :3] @ current[:3, :3].T)
        error = np.concatenate((position_mm / 1000.0, orientation))
        return error, float(np.linalg.norm(position_mm)), float(np.linalg.norm(orientation))

    def _jacobian(self, joints: np.ndarray, current: np.ndarray) -> np.ndarray:
        jacobian = np.empty((6, 6), dtype=float)
        for index in range(6):
            perturbed = joints.copy()
            perturbed[index] += self.difference_step_rad
            moved = self.forward(perturbed)
            jacobian[:3, index] = (
                (moved[:3, 3] - current[:3, 3]) / 1000.0 / self.difference_step_rad
            )
            jacobian[3:, index] = rotation_vector(
                moved[:3, :3] @ current[:3, :3].T
            ) / self.difference_step_rad
        return jacobian

    def solve(
        self,
        target_pose: np.ndarray,
        initial_joints: np.ndarray,
        lower_limits: np.ndarray | None = None,
        upper_limits: np.ndarray | None = None,
    ) -> LocalIKResult:
        target = pose6_transform(np.asarray(target_pose, dtype=float))
        joints = np.asarray(initial_joints, dtype=float).copy()
        lower = None if lower_limits is None else np.asarray(lower_limits, dtype=float)
        upper = None if upper_limits is None else np.asarray(upper_limits, dtype=float)

        if joints.shape != (6,) or not np.all(np.isfinite(joints)):
            return LocalIKResult(joints, False, 0, np.inf, np.inf, "invalid initial joints")

        for iteration in range(self.max_iterations + 1):
            current = self.forward(joints)
            error, position_error, orientation_error = self._error(target, current)
            if position_error <= self.position_tolerance_mm and orientation_error <= self.orientation_tolerance_rad:
                return LocalIKResult(
                    joints,
                    True,
                    iteration,
                    position_error,
                    float(np.rad2deg(orientation_error)),
                    "converged",
                )
            if iteration == self.max_iterations:
                break

            jacobian = self._jacobian(joints, current)
            system = jacobian @ jacobian.T + self.damping**2 * np.eye(6)
            try:
                step = jacobian.T @ np.linalg.solve(system, error)
            except np.linalg.LinAlgError:
                return LocalIKResult(
                    joints, False, iteration, position_error, float(np.rad2deg(orientation_error)), "singular solve"
                )
            if not np.all(np.isfinite(step)):
                return LocalIKResult(
                    joints, False, iteration, position_error, float(np.rad2deg(orientation_error)), "non-finite step"
                )
            step = np.clip(step, -self.max_iteration_step_rad, self.max_iteration_step_rad)

            # Backtracking prevents a bad linearization from increasing error.
            old_cost = float(error @ error)
            accepted = False
            scale = 1.0
            for _ in range(6):
                candidate = joints + scale * step
                if lower is not None:
                    candidate = np.maximum(candidate, lower)
                if upper is not None:
                    candidate = np.minimum(candidate, upper)
                candidate_error, _, _ = self._error(target, self.forward(candidate))
                if float(candidate_error @ candidate_error) < old_cost:
                    joints = candidate
                    accepted = True
                    break
                scale *= 0.5
            if not accepted:
                return LocalIKResult(
                    joints, False, iteration, position_error, float(np.rad2deg(orientation_error)), "no improving step"
                )

        current = self.forward(joints)
        _, position_error, orientation_error = self._error(target, current)
        return LocalIKResult(
            joints,
            False,
            self.max_iterations,
            position_error,
            float(np.rad2deg(orientation_error)),
            "iteration limit",
        )
