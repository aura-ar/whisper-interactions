import tempfile
import unittest
from pathlib import Path

import numpy as np

from contextual_bandit import (
    MAX_NUDGES,
    NUDGE_TYPES,
    SANDWICH_GOALS,
    ContextualCombinatorialThompsonSampling,
    reward_for_attempt,
)


class ContextualCombinatorialThompsonSamplingTests(unittest.TestCase):
    def test_selects_pair_without_reusing_either_factor(self):
        policy = ContextualCombinatorialThompsonSampling(
            random_seed=7,
        )

        first_decision = policy.select({"stage": "sandwich_choice", "attempt": 1})
        second_decision = policy.select(
            {
                "stage": "sandwich_choice",
                "attempt": 2,
                "previous_choice": "BLT",
            },
            excluded_nudges=(first_decision.action.nudge_type,),
            excluded_goals=(first_decision.action.sandwich_goal, "BLT"),
        )

        self.assertIn(first_decision.action.nudge_type, NUDGE_TYPES)
        self.assertIn(first_decision.action.sandwich_goal, SANDWICH_GOALS)
        self.assertNotEqual(
            second_decision.action.nudge_type,
            first_decision.action.nudge_type,
        )
        self.assertNotEqual(
            second_decision.action.sandwich_goal,
            first_decision.action.sandwich_goal,
        )
        self.assertNotEqual(second_decision.action.sandwich_goal, "BLT")

    def test_joint_reward_updates_posterior(self):
        policy = ContextualCombinatorialThompsonSampling(
            random_seed=2,
        )
        decision = policy.select({"stage": "sandwich_choice"})
        features = policy._features(decision.context, decision.action)

        policy.observe(decision, reward=1.0)
        posterior_mean = np.linalg.solve(
            policy._precision,
            policy._reward_features,
        )

        self.assertGreater(float(features @ posterior_mean), 0.0)
        self.assertEqual(policy.observation_count, 1)

    def test_reward_decreases_with_later_success(self):
        self.assertEqual(
            reward_for_attempt(switched_to_goal=True, attempt_number=1),
            1.0,
        )
        self.assertEqual(
            reward_for_attempt(switched_to_goal=True, attempt_number=2),
            0.75,
        )
        self.assertEqual(
            reward_for_attempt(switched_to_goal=True, attempt_number=3),
            0.5,
        )
        self.assertEqual(
            reward_for_attempt(switched_to_goal=False, attempt_number=2),
            -0.25,
        )

    def test_observation_persists_across_policy_instances(self):
        with tempfile.TemporaryDirectory() as directory:
            state_path = Path(directory) / "state.json"
            policy = ContextualCombinatorialThompsonSampling(
                state_path=state_path,
                random_seed=5,
            )
            decision = policy.select({"previous_choice": "BLT"})
            policy.observe(decision, reward=1.0)

            restored = ContextualCombinatorialThompsonSampling(
                state_path=state_path,
            )

            self.assertEqual(restored.observation_count, 1)
            self.assertEqual(restored.nudge_types, NUDGE_TYPES)
            self.assertEqual(restored.sandwich_goals, SANDWICH_GOALS)

    def test_attempt_number_is_bounded(self):
        with self.assertRaises(ValueError):
            reward_for_attempt(
                switched_to_goal=False,
                attempt_number=MAX_NUDGES + 1,
            )

    def test_exhausted_action_groups_raise(self):
        policy = ContextualCombinatorialThompsonSampling()

        with self.assertRaises(ValueError):
            policy.select(
                {},
                excluded_nudges=NUDGE_TYPES,
                excluded_goals=SANDWICH_GOALS,
            )


if __name__ == "__main__":
    unittest.main()