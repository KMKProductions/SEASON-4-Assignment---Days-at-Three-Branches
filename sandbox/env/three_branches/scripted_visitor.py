"""A social scripted visitor for Days at Three Branches."""

from __future__ import annotations

import math
import random
import re
from collections.abc import Mapping, Sequence
from typing import Any, cast


def _number(value: object, default: float = 0.0) -> float:
    try:
        return float(cast(Any, value))
    except (TypeError, ValueError):
        return default


def _position(record: Mapping[str, Any]) -> tuple[float, float]:
    point = record.get("position")
    if not isinstance(point, Mapping):
        return 0.0, 0.0
    return _number(point.get("x")), _number(point.get("y"))


def _heading_to(start: tuple[float, float], end: tuple[float, float]) -> float:
    return math.degrees(math.atan2(end[1] - start[1], end[0] - start[0])) % 360.0


_NPC_PLAYER_ID = re.compile(r"player_[1-9][0-9]*\Z")


def _self_record(observation: object) -> Mapping[str, Any]:
    if isinstance(observation, Mapping):
        record = observation.get("self")
        if isinstance(record, Mapping):
            return record
    return {}


def _seen_people(observation: object) -> list[Mapping[str, Any]]:
    if not isinstance(observation, Mapping):
        return []
    seen = observation.get("seen")
    if not isinstance(seen, Sequence) or isinstance(seen, str | bytes):
        return []
    return [person for person in seen if isinstance(person, Mapping) and isinstance(person.get("id"), str)]


class Agent:
    """Wander, greet a seen villager, linger, then continue through the village."""

    def reset(self, seed: object, observation: object) -> None:
        """Start a fresh visit. Builtins deliberately do not use the session seed."""
        del seed, observation
        self._rng = random.Random()
        self._mode = "wander"
        self._heading = self._rng.uniform(0.0, 360.0)
        self._remaining = self._rng.randint(18, 36)
        self._target: str | None = None
        self._linger = 0
        self._greeting_due = False
        self._replied = False

    def act(self, observation: object) -> dict[str, float | int]:
        """Choose a social movement and expression from the current observation."""
        me = _self_record(observation)
        position = _position(me)
        seen = _seen_people(observation)
        target = self._find_target(seen, position)

        if self._mode == "wander" and target is not None:
            self._target = str(target["id"])
            self._mode = "approach"
            self._replied = False

        if self._mode == "approach":
            if target is None:
                self._start_wandering()
            else:
                self._heading = _heading_to(position, _position(target))
                distance = math.dist(position, _position(target))
                if distance > 1.7:
                    return {"heading": self._heading, "speed": 0.8, "action": 0}
                self._mode = "linger"
                self._linger = self._rng.randint(4, 7)
                self._greeting_due = True
                return {"heading": self._heading, "speed": 0.0, "action": 2}

        if self._mode == "linger":
            self._linger -= 1
            if self._linger <= 0:
                self._mode = "move_on"
                self._remaining = self._rng.randint(12, 22)
                self._heading = (self._heading + self._rng.choice((135.0, 180.0, 225.0))) % 360.0
            return {"heading": self._heading, "speed": 0.0, "action": 0}

        if self._mode == "move_on":
            self._remaining -= 1
            if self._remaining <= 0:
                self._start_wandering()
            return {"heading": self._heading, "speed": 0.75, "action": 0}

        self._remaining -= 1
        if self._remaining <= 0:
            self._heading = (self._heading + self._rng.choice((-90.0, -60.0, 60.0, 90.0))) % 360.0
            self._remaining = self._rng.randint(18, 36)
        return {"heading": self._heading, "speed": 0.7, "action": 0}

    def chat(self, inbox: object) -> list[dict[str, str]]:
        """Offer one canned greeting, then one short reply to the same villager."""
        if self._target is None:
            return []
        if self._greeting_due:
            self._greeting_due = False
            return [{"to": self._target, "text": "Hello. How is your day going?"}]
        if self._replied or not isinstance(inbox, Sequence) or isinstance(inbox, str | bytes):
            return []
        for message in inbox:
            if isinstance(message, Mapping) and message.get("from") == self._target:
                self._replied = True
                return [{"to": self._target, "text": "Thank you. I am glad to be here."}]
        return []

    def _find_target(
        self, seen: list[Mapping[str, Any]], position: tuple[float, float]
    ) -> Mapping[str, Any] | None:
        """Keep pursuing one villager, or choose the nearest newly seen villager."""
        candidates = [person for person in seen if _NPC_PLAYER_ID.fullmatch(str(person.get("id", "")))]
        if not candidates:
            return None
        for person in candidates:
            if person.get("id") == self._target:
                return person
        return min(candidates, key=lambda person: math.dist(position, _position(person)))

    def _start_wandering(self) -> None:
        self._mode = "wander"
        self._target = None
        self._heading = (self._heading + self._rng.choice((-105.0, -75.0, 75.0, 105.0))) % 360.0
        self._remaining = self._rng.randint(18, 36)
