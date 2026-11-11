from __future__ import annotations

import json
import math
import os
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


NUDGE_TYPES = ("Social Influence", "Warning", "Anchoring")
SANDWICH_GOALS = ("BLT", "Chicken", "Prawn")
MAX_NUDGES = 3


@dataclass(frozen=True)
class BanditAction:
    nudge_type: str
    sandwich_goal: str


@dataclass(frozen=True)
class BanditDecision:
    decision_id: str
    context: dict[str, Any]
    action: BanditAction


def reward_for_attempt(
    *,
    switched_to_goal: bool,
    attempt_number: int,
    penalty_per_extra_attempt: float = 0.25,
    penalty_per_unsuccessful_attempt: float = 0.25,
) -> float:
    if not 1 <= attempt_number <= MAX_NUDGES:
        raise ValueError(f"attempt_number must be between 1 and {MAX_NUDGES}")
    if switched_to_goal:
        return 1.0 - penalty_per_extra_attempt * (attempt_number - 1)
    return -penalty_per_unsuccessful_attempt


class ContextualCombinatorialThompsonSampling:
    """Factorized Thompson sampling with joint reward feedback for action pairs."""

    def __init__(
        self,
        *,
        nudge_types: tuple[str, ...] = NUDGE_TYPES,
        sandwich_goals: tuple[str, ...] = SANDWICH_GOALS,
        exploration_scale: float = 0.5,
        prior_precision: float = 1.0,
        state_path: Path | None = None,
        random_seed: int | None = None,
    ) -> None:
        if not nudge_types or not sandwich_goals:
            raise ValueError("Both action groups must have at least one value")
        if exploration_scale < 0 or prior_precision <= 0:
            raise ValueError(
                "exploration_scale must be non-negative, prior_precision positive, "
                "and feature groups non-empty"
            )

        self.nudge_types = tuple(nudge_types)
        self.sandwich_goals = tuple(sandwich_goals)
        self.exploration_scale = exploration_scale
        self.prior_precision = prior_precision
        self._context_dimension = (
            1 + 2 + 4 + len(sandwich_goals) + 2 + len(nudge_types) + len(sandwich_goals)
        )
        self.feature_dimension = self._context_dimension * (
            1 + len(nudge_types) + len(sandwich_goals) + len(nudge_types) * len(sandwich_goals)
        )
        self.state_path = state_path
        self._precision = np.eye(self.feature_dimension, dtype=np.float64) * prior_precision
        self._reward_features = np.zeros(self.feature_dimension, dtype=np.float64)
        self._rng = np.random.default_rng(random_seed)
        self.observation_count = 0

        if state_path is not None and state_path.exists():
            self._load()

    def select(
        self,
        context: dict[str, Any],
        *,
        excluded_nudges: tuple[str, ...] = (),
        excluded_goals: tuple[str, ...] = (),
    ) -> BanditDecision:
        candidates = [
            BanditAction(nudge_type, sandwich_goal)
            for nudge_type in self.nudge_types
            if nudge_type not in excluded_nudges
            for sandwich_goal in self.sandwich_goals
            if sandwich_goal not in excluded_goals
        ]
        if not candidates:
            raise ValueError("No valid action pairs remain for the supplied context")

        posterior_mean = np.linalg.solve(self._precision, self._reward_features)
        cholesky = np.linalg.cholesky(self._precision)
        posterior_sample = posterior_mean + self.exploration_scale * np.linalg.solve(
            cholesky.T,
            self._rng.standard_normal(self.feature_dimension),
        )
        action = max(
            candidates,
            key=lambda candidate: float(
                self._features(context, candidate) @ posterior_sample
            ),
        )
        return BanditDecision(
            decision_id=str(uuid.uuid4()),
            context=dict(context),
            action=action,
        )

    def observe(self, decision: BanditDecision, reward: float) -> None:
        if not math.isfinite(reward):
            raise ValueError("reward must be finite")
        if decision.action.nudge_type not in self.nudge_types:
            raise ValueError("decision contains an unknown nudge type")
        if decision.action.sandwich_goal not in self.sandwich_goals:
            raise ValueError("decision contains an unknown sandwich goal")

        features = self._features(decision.context, decision.action)
        self._precision += np.outer(features, features)
        self._reward_features += reward * features
        self.observation_count += 1
        if self.state_path is not None:
            self._save()

    def _features(self, context: dict[str, Any], action: BanditAction) -> np.ndarray:
        context_features = self._context_features(context)
        nudge_features = np.zeros(len(self.nudge_types), dtype=np.float64)
        goal_features = np.zeros(len(self.sandwich_goals), dtype=np.float64)
        nudge_features[self.nudge_types.index(action.nudge_type)] = 1.0
        goal_features[self.sandwich_goals.index(action.sandwich_goal)] = 1.0
        pair_features = np.outer(nudge_features, goal_features).ravel()

        return np.concatenate(
            (
                context_features,
                np.kron(nudge_features, context_features),
                np.kron(goal_features, context_features),
                np.kron(pair_features, context_features),
            )
        )

    def _context_features(self, context: dict[str, Any]) -> np.ndarray:
        features = np.zeros(self._context_dimension, dtype=np.float64)
        index = 0

        features[index] = 1.0
        index += 1

        stage = context.get("stage")
        features[index + (0 if stage == "sandwich_choice" else 1)] = 1.0
        index += 2

        attempt = context.get("attempt")
        attempt_index = attempt - 1 if attempt in (1, 2, 3) else 3
        features[index + attempt_index] = 1.0
        index += 4

        previous_choice = context.get("previous_choice")
        previous_index = (
            self.sandwich_goals.index(previous_choice) + 1
            if previous_choice in self.sandwich_goals
            else 0 if previous_choice is None else len(self.sandwich_goals) + 1
        )
        features[index + previous_index] = 1.0
        index += len(self.sandwich_goals) + 2

        used_nudges = set(context.get("used_nudges", ()))
        for nudge_index, nudge in enumerate(self.nudge_types):
            features[index + nudge_index] = float(nudge in used_nudges)
        index += len(self.nudge_types)

        used_goals = set(context.get("used_goals", ()))
        for goal_index, goal in enumerate(self.sandwich_goals):
            features[index + goal_index] = float(goal in used_goals)

        return features

    def _save(self) -> None:
        assert self.state_path is not None
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": 2,
            "nudge_types": self.nudge_types,
            "sandwich_goals": self.sandwich_goals,
            "exploration_scale": self.exploration_scale,
            "prior_precision": self.prior_precision,
            "feature_dimension": self.feature_dimension,
            "precision": self._precision.tolist(),
            "reward_features": self._reward_features.tolist(),
            "observation_count": self.observation_count,
        }
        temporary_path: str | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=self.state_path.parent,
                delete=False,
            ) as file:
                json.dump(payload, file)
                temporary_path = file.name
            os.replace(temporary_path, self.state_path)
        finally:
            if temporary_path is not None and os.path.exists(temporary_path):
                os.unlink(temporary_path)

    def _load(self) -> None:
        assert self.state_path is not None
        with self.state_path.open(encoding="utf-8") as file:
            payload = json.load(file)
        if payload.get("schema_version") != 2:
            raise ValueError(
                "Saved CCB state uses an older policy format; back it up and "
                "remove it to initialize the Thompson-sampling policy"
            )
        if tuple(payload["nudge_types"]) != self.nudge_types:
            raise ValueError("Saved CCB state uses a different nudge action set")
        if tuple(payload["sandwich_goals"]) != self.sandwich_goals:
            raise ValueError("Saved CCB state uses a different sandwich action set")
        if payload["feature_dimension"] != self.feature_dimension:
            raise ValueError("Saved CCB state uses a different feature dimension")
        if payload["prior_precision"] != self.prior_precision:
            raise ValueError("Saved CCB state uses a different prior precision")
        self._precision = np.asarray(payload["precision"], dtype=np.float64)
        self._reward_features = np.asarray(
            payload["reward_features"],
            dtype=np.float64,
        )
        self.observation_count = int(payload["observation_count"])