"""Days at Three Branches villagers built from small, reusable behaviour blocks.

Every villager has one role. A role is a tree of behaviour blocks: each block's ``tick`` returns
the next order while it is running, or ``None`` once it has finished so its parent can move on.
The platform runs a separate ``Agent`` instance for every villager, so each instance builds only
its own tree in ``reset``.
"""

from __future__ import annotations

import heapq
import math
import re
from collections import deque
from collections.abc import Callable, Iterable
from dataclasses import dataclass, field

from sandbox.observation_types import ThreeBranchesAction, ThreeBranchesObservation
from sandbox.village import action, day, geometry, layout, me, people, props

# --- Roles -------------------------------------------------------------------------------------

BLACKSMITH_IDS = ("player_1", "player_2")
PRIEST_IDS = ("player_3", "player_4")
GARDENER_IDS = ("player_5", "player_6")
WANDERER_IDS = ("player_7", "player_8", "player_9")
SLEEPER_IDS = ("player_10",)
VISITOR_ID = "player_0"
WORKER_IDS = (*BLACKSMITH_IDS, *PRIEST_IDS, *GARDENER_IDS)

ROLES = {
    **{player_id: "blacksmith" for player_id in BLACKSMITH_IDS},
    **{player_id: "priest" for player_id in PRIEST_IDS},
    **{player_id: "gardener" for player_id in GARDENER_IDS},
    **{player_id: "wanderer" for player_id in WANDERER_IDS},
    **{player_id: "sleeper" for player_id in SLEEPER_IDS},
}

# --- Tunables (ticks unless noted) -------------------------------------------------------------

STALL_WORK_TICKS = 60
REPAIR_BENCH_WORK_TICKS = 80
PUMP_TICKS = 12
SHRINE_TICKS = 120
PLOT_WORK_TICKS = 70
GREET_TICKS = 8
OBSERVE_TICKS = 60
OBSERVE_DISTANCE = 2.5  # metres a wanderer keeps from the worker it watches
CHAT_TICKS = 40
CHAT_DISTANCE = 1.8  # metres between two chatting wanderers
CHAT_SEE_RANGE = 6.0  # metres: a wanderer seen this close starts a chat
CHAT_HEAR_RANGE = 3.0  # metres: a wanderer heard this close starts a chat
CHAT_COOLDOWN_TICKS = 250
CHAT_EMOTES = ("wave", "laugh", "nod", "point", "shrug", "laugh", "nod")
LOST_TICKS = 20  # how long a wanderer looks for a vanished worker before moving on
STUCK_TICKS = 6  # ticks without movement before a walker re-plans around people
NUDGE_TICKS = 10  # ticks spent shuffling closer when a prop is not yet usable
CLEARANCE_CAP = 3  # cells: open ground beyond this far from obstacles all counts the same
WALL_PENALTY = (0.8, 0.25, 0.0)  # extra route cost for a cell 0, 1, or 2+ cells from an obstacle
LINE_SAMPLE_STEP = 0.25  # metres between the points checked along a straight-line shortcut
LOOKAHEAD_CELLS = 8  # how far along its route a walker looks for a straight-line shortcut
AVOID_DISTANCE = 1.6  # metres: someone this close and ahead makes a walker veer aside
AVOID_ANGLE = 35.0  # degrees either side of the walking direction that count as "ahead"
AVOID_TURN = 40.0  # degrees a walker veers to pass someone
AVOID_COMMIT_TICKS = 4  # keep veering the same way this long, so a walker doesn't wobble
STUCK_PROGRESS = 0.2  # metres: moving less than this in a tick counts as stuck
REACH_MARGIN = 0.35  # stay this far inside use reach when choosing a standing cell
BREAK_TICKS = 100  # how long a worker sits on a bench after each full work cycle
BREAK_BENCH_TRIES = 3  # benches a worker tries before resting on their feet instead
BREAK_CHAT_TICKS = 60  # how long a wanderer chats with a worker on break
BREAK_CHAT_LINE_TICKS = 15  # ticks between a wanderer's lines of small talk
BREAK_CHAT_RANGE = 10.0  # metres: a wanderer notices a worker on break this close
BREAK_CHAT_COOLDOWN_TICKS = 250
BREAK_CHAT_EMOTES = ("nod", "laugh", "point", "shrug")
SMALL_TALK = (
    "How's the work going today?",
    "The village feels busy this morning.",
    "Have you heard the bell lately?",
    "Well, I'll let you rest. Enjoy your break!",
)
BREAK_REPLIES = {
    "blacksmith": (
        "Ah, the muscles need a rest.",
        "It's tough work at the forge.",
        "Busy, busy. The stalls need tending all day.",
        "Thanks, friend. Back to the forge soon.",
    ),
    "priest": (
        "Ah, it's good to sit down for a bit.",
        "I feel a divine presence today. A blessing indeed.",
        "Every day is a blessing, busy or not.",
        "Thank you. May your day continue to be well.",
    ),
    "gardener": (
        "Ah, just need a little rest.",
        "Those darn weeds never stop growing!",
        "Lucky us, the soil's good this year.",
        "Thanks! The gardens won't tend themselves.",
    ),
}
HELLO_COOLDOWN_TICKS = 200  # after greeting someone, ignore their hellos this long (stops echo loops)
HELLO_WAVE_TICKS = 6  # how long a villager waves back
HELLO_WAVE_RANGE = 8.0  # metres: a wave from further away is not meant for us
HELLO_WAVE_ANGLE = 60.0  # degrees: a waver must be facing within this of us
HELLO_WORDS = re.compile(r"\b(hello|hi|hey|hiya|howdy|greetings|good (morning|afternoon|evening|day))\b", re.I)
PLAYER_ID_PATTERN = re.compile(r"\bplayer_\d+\b")
# What villagers call each other in chat: by role, never by player id.
ROLE_TITLES = {
    "blacksmith": "blacksmith",
    "priest": "priest",
    "gardener": "gardener",
    "wanderer": "wanderer",
    "sleeper": "sleepyhead",
}
VISITOR_TITLE = "traveler"
TITLE_PATTERN = re.compile(r"\b(" + "|".join((*ROLE_TITLES.values(), VISITOR_TITLE)) + r")\b", re.I)
HELLO_REPLIES = {
    "blacksmith": "Hello, {name}! Can't stop, it's hammer time!.",
    "priest": "Hello, {name}! Blessings on this fine day.",
    "gardener": "Hello, {name}! It's gonna be a good harvest today, I can feel it!.",
    "wanderer": "Hello, {name}! Lovely day for a stroll.",
    "sleeper": "Zzz... mm, hello {name}... zzz.",
    "sleeper_awake": "Hello, {name}! I'm finally up and about.",
}
WAKE_TICKS = 10  # how long a startled sleeper jumps before setting off
WAKE_LINE = "Wha-?! Who's there? Alright, alright, I'm up!"


def title(player_id: str) -> str:
    """What villagers call someone in chat: their role, or "traveler" for the visitor."""

    if people.is_visitor(player_id):
        return VISITOR_TITLE
    return ROLE_TITLES.get(ROLES.get(player_id, ""), "friend")

# Prop footprints in cells, (width, height) before an east/west facing swaps them.
PROP_FOOTPRINTS = {"bench": (2, 1), "stall": (2, 2), "repair_bench": (2, 1), "shrine": (3, 3), "plot": (4, 2), "pump": (1, 1)}
CIRCLE_PROPS = {"pump"}

Cell = tuple[int, int]


# --- Navigation --------------------------------------------------------------------------------

_NAV_CACHE: dict[tuple, Navigator] = {}

CARDINALS = ((1, 0), (-1, 0), (0, 1), (0, -1))
DIAGONALS = ((1, 1), (1, -1), (-1, 1), (-1, -1))


def _village_key(observation: ThreeBranchesObservation) -> tuple:
    """Identify a village layout so instances in one process can share its navigator."""

    spawn = layout.spawn(observation)
    placements = tuple(
        (prop["id"], int(prop["cell"]["x"]), int(prop["cell"]["y"])) for prop in props.all(observation)
    )
    return float(spawn["x"]), float(spawn["y"]), placements


class Navigator:
    """The village as a weighted walking graph, built once per day.

    Moves go in eight directions; a diagonal is allowed only when both L-shaped detours around it
    are legal, so it never clips a corner. Each move costs its length, scaled by how slow the
    ground is (so roads and paths are preferred over fields and reeds), plus a penalty for cells
    hugging a wall, prop, or tree. Routes therefore keep a little room from obstacles."""

    def __init__(self, observation: ThreeBranchesObservation):
        size = layout.frame(observation)
        self.cell_size = float(size["cell_size"])
        cells_x, cells_y = int(size["cells_x"]), int(size["cells_y"])
        walkable = {
            (x, y) for x in range(cells_x) for y in range(cells_y) if layout.walkable(observation, {"x": x, "y": y})
        }
        steps: dict[Cell, set[Cell]] = {}
        for x, y in walkable:
            here = {"x": x, "y": y}
            steps[(x, y)] = {
                (x + dx, y + dy)
                for dx, dy in CARDINALS
                if (x + dx, y + dy) in walkable and layout.can_step(observation, here, {"x": x + dx, "y": y + dy})
            }

        # Clearance: how many cells of open ground surround a cell before the nearest obstacle.
        self.clearance: dict[Cell, int] = {}
        frontier = deque()
        for cell in walkable:
            x, y = cell
            if any((x + dx, y + dy) not in walkable for dx, dy in (*CARDINALS, *DIAGONALS)):
                self.clearance[cell] = 0
                frontier.append(cell)
        while frontier:
            x, y = frontier.popleft()
            depth = self.clearance[(x, y)] + 1
            if depth > CLEARANCE_CAP:
                continue
            for dx, dy in (*CARDINALS, *DIAGONALS):
                neighbour = (x + dx, y + dy)
                if neighbour in walkable and neighbour not in self.clearance:
                    self.clearance[neighbour] = depth
                    frontier.append(neighbour)

        slowness = {}
        for cell in walkable:
            ground = layout.ground_at(observation, {"x": cell[0], "y": cell[1]})
            speed = layout.SPEED_LIMITS.get(ground or "", 1.0) or 1.0
            slowness[cell] = 1.0 / speed

        self.edges: dict[Cell, tuple[tuple[Cell, float], ...]] = {}
        for cell, cardinal in steps.items():
            x, y = cell
            moves = [(neighbour, 1.0) for neighbour in cardinal]
            for dx, dy in DIAGONALS:
                across, side_x, side_y = (x + dx, y + dy), (x + dx, y), (x, y + dy)
                if (
                    side_x in cardinal
                    and side_y in cardinal
                    and across in steps.get(side_x, ())
                    and across in steps.get(side_y, ())
                ):
                    moves.append((across, 1.41421356))
            self.edges[cell] = tuple(
                (
                    neighbour,
                    length * (slowness[cell] + slowness[neighbour]) / 2
                    + WALL_PENALTY[min(self.clearance.get(neighbour, CLEARANCE_CAP), len(WALL_PENALTY) - 1)],
                )
                for neighbour, length in moves
            )

    @classmethod
    def for_village(cls, observation: ThreeBranchesObservation) -> Navigator:
        key = _village_key(observation)
        navigator = _NAV_CACHE.get(key)
        if navigator is None:
            navigator = _NAV_CACHE[key] = cls(observation)
        return navigator

    def __contains__(self, cell: Cell) -> bool:
        return cell in self.edges

    def route(self, start: Cell, goals: frozenset[Cell], blocked: frozenset[Cell] = frozenset()) -> list[Cell] | None:
        """A* search for the cheapest route from ``start`` to any goal cell."""

        if start in goals:
            return [start]
        low_x = min(x for x, _ in goals)
        high_x = max(x for x, _ in goals)
        low_y = min(y for _, y in goals)
        high_y = max(y for _, y in goals)

        def estimate(cell: Cell) -> float:
            # Octile distance to the goals' bounding box: never more than the true cost.
            dx = max(low_x - cell[0], 0, cell[0] - high_x)
            dy = max(low_y - cell[1], 0, cell[1] - high_y)
            return max(dx, dy) + 0.41421356 * min(dx, dy)

        cost: dict[Cell, float] = {start: 0.0}
        previous: dict[Cell, Cell | None] = {start: None}
        queue = [(estimate(start), 0, start)]
        order = 0
        while queue:
            _, _, current = heapq.heappop(queue)
            if current in goals:
                route = []
                step: Cell | None = current
                while step is not None:
                    route.append(step)
                    step = previous[step]
                route.reverse()
                return route
            base = cost[current]
            for neighbour, weight in self.edges.get(current, ()):
                if neighbour in blocked and neighbour not in goals:
                    continue
                total = base + weight
                if total < cost.get(neighbour, float("inf")):
                    cost[neighbour] = total
                    previous[neighbour] = current
                    order += 1
                    heapq.heappush(queue, (total + estimate(neighbour), order, neighbour))
        return None

    def line_clear(self, start, end) -> bool:
        """Whether a villager can walk straight from ``start`` to ``end``: every point on the way
        lies in open ground with room to spare (cells next to an obstacle only count at the two
        ends, where the route itself already goes)."""

        sx, sy = float(start["x"]), float(start["y"])
        ex, ey = float(end["x"]), float(end["y"])
        length = ((ex - sx) ** 2 + (ey - sy) ** 2) ** 0.5
        samples = max(1, int(length / LINE_SAMPLE_STEP))
        size = self.cell_size
        first = (int(sx // size), int(sy // size))
        last = (int(ex // size), int(ey // size))
        for index in range(samples + 1):
            fraction = index / samples
            cell = (int((sx + (ex - sx) * fraction) // size), int((sy + (ey - sy) * fraction) // size))
            if cell not in self.edges:
                return False
            if cell not in (first, last) and self.clearance.get(cell, 0) < 1:
                return False
        return True

    def centre(self, cell: Cell) -> dict[str, float]:
        return {"x": (cell[0] + 0.5) * self.cell_size, "y": (cell[1] + 0.5) * self.cell_size}


def _prop_shape(observation: ThreeBranchesObservation, prop) -> tuple[str, float, float, float, float]:
    """Return a prop's collision shape as ``(kind, x, y, width, height)`` in metres."""

    cell_size = float(layout.frame(observation)["cell_size"])
    width, height = PROP_FOOTPRINTS.get(str(prop["type"]), (1, 1))
    if prop.get("facing") in {"east", "west"}:
        width, height = height, width
    x, y = int(prop["cell"]["x"]) * cell_size, int(prop["cell"]["y"]) * cell_size
    width, height = width * cell_size, height * cell_size
    if prop["type"] in CIRCLE_PROPS:
        diameter = min(width, height)
        return "circle", x + (width - diameter) / 2, y + (height - diameter) / 2, diameter, diameter
    return "box", x, y, width, height


def _prop_center(observation: ThreeBranchesObservation, prop) -> dict[str, float]:
    _, x, y, width, height = _prop_shape(observation, prop)
    return {"x": x + width / 2, "y": y + height / 2}


def _nearest_on_prop(observation: ThreeBranchesObservation, prop, point) -> tuple[dict[str, float], float]:
    """Return the closest point of a prop's shape to ``point`` and the distance to it."""

    kind, x, y, width, height = _prop_shape(observation, prop)
    px, py = float(point["x"]), float(point["y"])
    if kind == "circle":
        centre = {"x": x + width / 2, "y": y + height / 2}
        gap = geometry.distance(point, centre)
        radius = width / 2
        if gap == 0:
            return {"x": centre["x"] + radius, "y": centre["y"]}, 0.0
        scale = radius / gap
        nearest = {"x": centre["x"] + (px - centre["x"]) * scale, "y": centre["y"] + (py - centre["y"]) * scale}
        return nearest, max(0.0, gap - radius)
    nearest = {"x": min(max(px, x), x + width), "y": min(max(py, y), y + height)}
    return nearest, geometry.distance(point, nearest)


def _approach_cells(observation: ThreeBranchesObservation, graph, prop) -> frozenset[Cell]:
    """Return walkable cells whose centre is comfortably within use reach of a prop, with no wall
    in between, so a villager standing there can use it."""

    _, x, y, width, height = _prop_shape(observation, prop)
    reach = geometry.PROP_REACH
    low_x, low_y = int(x - reach) - 1, int(y - reach) - 1
    high_x, high_y = int(x + width + reach) + 1, int(y + height + reach) + 1
    for limit in (reach - REACH_MARGIN, reach):
        found = set()
        for cx in range(low_x, high_x + 1):
            for cy in range(low_y, high_y + 1):
                if (cx, cy) not in graph:
                    continue
                centre = layout.cell_center(observation, {"x": cx, "y": cy})
                nearest, gap = _nearest_on_prop(observation, prop, centre)
                if gap > limit:
                    continue
                # Test sight to a point just outside the prop edge, which is never inside a wall.
                if gap > 0:
                    pull = min(0.1, gap) / gap
                    nearest = {
                        "x": nearest["x"] + (centre["x"] - nearest["x"]) * pull,
                        "y": nearest["y"] + (centre["y"] - nearest["y"]) * pull,
                    }
                if layout.line_of_sight(observation, centre, nearest):
                    found.add((cx, cy))
        if found:
            return frozenset(found)
    return frozenset()


def _home_cells(observation: ThreeBranchesObservation, graph, building_id: str) -> frozenset[Cell]:
    """Return the walkable floor cells inside a building, skipping its wall ring."""

    building = layout.building(observation, building_id)
    if building is None:
        return frozenset()
    width, height = layout.BUILDING_SIZES[str(building["type"])]
    x0, y0 = int(building["cell"]["x"]), int(building["cell"]["y"])
    return frozenset(
        (x, y)
        for x in range(x0 + 1, x0 + width - 1)
        for y in range(y0 + 1, y0 + height - 1)
        if (x, y) in graph
    )


def _cells_near(graph, cell: Cell, radius: int) -> frozenset[Cell]:
    """Return walkable cells within a square ``radius`` of ``cell``, excluding the cell itself."""

    x0, y0 = cell
    return frozenset(
        (x, y)
        for x in range(x0 - radius, x0 + radius + 1)
        for y in range(y0 - radius, y0 + radius + 1)
        if (x, y) in graph and (x, y) != cell
    )


# --- Shared, deterministic work assignments ----------------------------------------------------


def _props_of(observation: ThreeBranchesObservation, prop_type: str) -> list[str]:
    return sorted(str(prop["id"]) for prop in props.all(observation) if prop["type"] == prop_type)


def _share(items: list[str], member_ids: tuple[str, ...], player_id: str) -> list[str]:
    """Split ``items`` round-robin between role members so each one owns its own share."""

    if not items or player_id not in member_ids:
        return []
    return items[member_ids.index(player_id) :: len(member_ids)]


def work_sites(observation: ThreeBranchesObservation, player_id: str) -> list[str]:
    """Return the ordered prop ids a worker visits. Every instance computes the same answer, which
    is how wanderers know where to look for a worker they cannot yet see."""

    role = ROLES.get(player_id)
    if role == "blacksmith":
        own = _share(_props_of(observation, "stall"), BLACKSMITH_IDS, player_id)
        repair = _props_of(observation, "repair_bench")[:1]
        # Opposite offsets keep the two blacksmiths from reaching the shared repair bench together.
        return repair + own if BLACKSMITH_IDS.index(player_id) == 0 else own + repair
    if role == "priest":
        pumps = _props_of(observation, "pump")[:1]
        shrines = _props_of(observation, "shrine")
        if not shrines:
            return pumps
        own = shrines[PRIEST_IDS.index(player_id) % len(shrines)]
        return [*pumps, own]
    if role == "gardener":
        return _share(_props_of(observation, "plot"), GARDENER_IDS, player_id)
    return []


# --- Greetings ---------------------------------------------------------------------------------


class Greetings:
    """One villager's hellos: who greeted us and still needs an answer, who we greeted recently,
    and the chat messages waiting to go out on our next ``chat`` turn."""

    def __init__(self, me_id: str):
        self.me_id = me_id
        self.pending: list[str] = []
        self.last_greeted: dict[str, int] = {}
        self.outbox: list[dict] = []
        self.hearers: frozenset[str] = frozenset()

    def heard_hello(self, from_id: str, tick: int) -> None:
        """Note a hello from someone, unless we greeted each other moments ago."""

        if from_id == self.me_id or from_id in self.pending:
            return
        if tick - self.last_greeted.get(from_id, -HELLO_COOLDOWN_TICKS) < HELLO_COOLDOWN_TICKS:
            return
        self.pending.append(from_id)

    def say_hello(self, to_id: str, text: str, tick: int) -> None:
        """Queue a chat hello. It goes direct when they can hear us, otherwise as a broadcast that
        names their role, so villagers in other roles don't mistake it for a hello of their own."""

        self.last_greeted[to_id] = tick
        self.say(to_id, text)

    def say(self, to_id: str, text: str) -> None:
        """Queue a chat line to one villager, routed like a hello but without its cooldown."""

        recipient = to_id if to_id in self.hearers else None
        self.outbox = [message for message in self.outbox if message["to"] != recipient]
        self.outbox.append({"to": recipient, "text": text})

    def is_hello_for_me(self, message: dict) -> bool:
        """Whether a chat message greets this villager: direct to us, or a broadcast that names
        nobody in particular, or names us by id or by our role."""

        text = str(message.get("text", ""))
        if not HELLO_WORDS.search(text):
            return False
        if message.get("to") == self.me_id:
            return True
        if message.get("to") is not None:
            return False
        ids = PLAYER_ID_PATTERN.findall(text)
        titles = {found.lower() for found in TITLE_PATTERN.findall(text)}
        if not ids and not titles:
            return True
        return self.me_id in ids or title(self.me_id) in titles


def spot_waves(ctx: Context) -> None:
    """Treat a nearby villager who waves while facing us as saying hello."""

    for person in people.seen(ctx.obs):
        if person["expression"]["type"] != "wave":
            continue
        gap = geometry.distance(ctx.here, person["position"])
        if gap > HELLO_WAVE_RANGE:
            continue
        toward_me = geometry.heading_to(person["position"], ctx.here)
        off = abs((float(person["heading"]) - toward_me + 180.0) % 360.0 - 180.0)
        if off <= HELLO_WAVE_ANGLE:
            ctx.greetings.heard_hello(str(person["id"]), ctx.tick)


# --- Behaviour blocks --------------------------------------------------------------------------


@dataclass
class Context:
    """Everything a block needs for one tick."""

    obs: ThreeBranchesObservation
    me_id: str
    here: dict[str, float]
    cell: Cell | None
    tick: int
    nav: Navigator
    approach: dict[str, frozenset[Cell]]
    greetings: Greetings
    memory: dict = field(default_factory=dict)

    @property
    def graph(self) -> Navigator:
        """The walkable cells, for membership tests such as ``cell in ctx.graph``."""

        return self.nav

    def stand(self, expression: str = "none") -> ThreeBranchesAction:
        return action.stand(me.heading(self.obs), expression)

    def face(self, point, expression: str = "none") -> ThreeBranchesAction:
        return action.stand(geometry.heading_to(self.here, point), expression)

    def approach_cells(self, prop_id: str) -> frozenset[Cell]:
        cells = self.approach.get(prop_id)
        if cells is None:
            prop = props.find(self.obs, prop_id)
            cells = frozenset() if prop is None else _approach_cells(self.obs, self.graph, prop)
            self.approach[prop_id] = cells
        return cells

    def people_cells(self) -> frozenset[Cell]:
        cells = set()
        for person in (*people.seen(self.obs), *people.nearby(self.obs)):
            if str(person["id"]) == self.me_id:
                continue
            found = layout.cell_at(self.obs, person["position"])
            if found is not None:
                cells.add((found["x"], found["y"]))
        return frozenset(cells)


class Block:
    """A behaviour block. ``tick`` returns an order while running and ``None`` once finished."""

    def tick(self, ctx: Context) -> ThreeBranchesAction | None:
        raise NotImplementedError


class GoTo(Block):
    """Walk a planned route to any cell from ``goals_fn``. Each tick it heads for the furthest
    point along the route it can reach in a clear straight line, so it walks smooth lines and
    diagonals instead of cell-by-cell zig-zags, and veers around people just ahead. Re-plans when
    knocked off the route, and around people when it stops making progress."""

    def __init__(self, goals_fn: Callable[[Context], Iterable[Cell]]):
        self.goals_fn = goals_fn
        self.goals: frozenset[Cell] | None = None
        self.route: list[Cell] | None = None
        self.progress = 0  # index of the furthest route cell reached so far
        self.veer = 0.0  # degrees we are currently veering around someone
        self.veer_left = 0
        self.stuck = 0
        self.failed = False

    def tick(self, ctx: Context) -> ThreeBranchesAction | None:
        if self.goals is None:
            self.goals = frozenset(self.goals_fn(ctx))
        if not self.goals or ctx.cell is None:
            self.failed = True
            return None
        if ctx.cell in self.goals:
            return None

        self.stuck = self.stuck + 1 if self.route is not None and me.moved(ctx.obs) < STUCK_PROGRESS else 0
        if self.route is None or not self._track(ctx.cell) or self.stuck >= STUCK_TICKS:
            blocked = ctx.people_cells() if self.stuck >= STUCK_TICKS else frozenset()
            route = ctx.nav.route(ctx.cell, self.goals, blocked)
            if route is None and blocked:
                route = ctx.nav.route(ctx.cell, self.goals)
            if route is None:
                self.failed = True
                return None
            self.route, self.progress, self.stuck = route, 0, 0

        heading = geometry.heading_to(ctx.here, self._aim(ctx))
        if self.veer_left > 0 and _open_ahead(ctx, heading + self.veer):
            self.veer_left -= 1
        else:
            self.veer = _avoid_people(ctx, heading)
            self.veer_left = AVOID_COMMIT_TICKS if self.veer else 0
        return action.walk(geometry.wrap(heading + self.veer), 1.0, "none")

    def _track(self, cell: Cell) -> bool:
        """Advance progress to where we are on the route. Shortcuts may leave us beside the route
        rather than on it; only straying further than one cell away counts as being off route."""

        window = self.route[self.progress : self.progress + LOOKAHEAD_CELLS + 2]
        for offset in range(len(window) - 1, -1, -1):
            if window[offset] == cell:
                self.progress += offset
                return True
        return any(max(abs(cell[0] - x), abs(cell[1] - y)) <= 1 for x, y in window[:4])

    def _aim(self, ctx: Context) -> dict[str, float]:
        last = len(self.route) - 1
        for index in range(min(self.progress + LOOKAHEAD_CELLS, last), self.progress, -1):
            point = ctx.nav.centre(self.route[index])
            if ctx.nav.line_clear(ctx.here, point):
                return point
        return ctx.nav.centre(self.route[min(self.progress + 1, last)])


def _open_ahead(ctx: Context, heading: float) -> bool:
    """Whether the next stride in this direction is open ground."""

    ahead = {
        "x": ctx.here["x"] + 0.8 * math.cos(math.radians(heading)),
        "y": ctx.here["y"] + 0.8 * math.sin(math.radians(heading)),
    }
    return ctx.nav.line_clear(ctx.here, ahead)


def _avoid_people(ctx: Context, heading: float) -> float:
    """Return how many degrees to veer to pass someone just ahead (0 when the way is clear or no
    side step is open ground)."""

    for person in people.seen(ctx.obs):
        if str(person["id"]) == ctx.me_id:
            continue
        gap = geometry.distance(ctx.here, person["position"])
        if gap > AVOID_DISTANCE:
            continue
        bearing = geometry.heading_to(ctx.here, person["position"])
        off = (bearing - heading + 180.0) % 360.0 - 180.0
        if abs(off) > AVOID_ANGLE:
            continue
        # Someone is ahead: try veering away from their side first, then the other way.
        for turn in ((-AVOID_TURN, AVOID_TURN) if off >= 0 else (AVOID_TURN, -AVOID_TURN)):
            if _open_ahead(ctx, heading + turn):
                return turn
        break
    return 0.0


class UseProp(Block):
    """Face a prop and hold ``use`` on it for ``ticks``. Shuffles closer first if the engine would
    not yet pick this prop, and gives up if it never becomes usable. ``blocked`` is set when it
    gives up because somebody else is holding the prop.

    ``face_fn`` may return a point to face instead of the prop: turning does not let go of it."""

    def __init__(
        self,
        prop_id: str,
        ticks: int,
        skip_if_busy: bool = False,
        face_fn: Callable[[Context], dict[str, float] | None] | None = None,
    ):
        self.prop_id = prop_id
        self.left = ticks
        self.skip_if_busy = skip_if_busy
        self.face_fn = face_fn
        self.started = False
        self.blocked = False
        self.misses = 0
        self.nudges = 0

    def tick(self, ctx: Context) -> ThreeBranchesAction | None:
        prop = props.find(ctx.obs, self.prop_id)
        if prop is None or self.left <= 0:
            return None
        centre = _prop_center(ctx.obs, prop)
        if not self.started and self.skip_if_busy and props.state(ctx.obs, self.prop_id) == "busy":
            return None

        usable = props.usable(ctx.obs)
        if usable is None or usable["id"] != self.prop_id:
            self.nudges += 1
            if self.nudges > NUDGE_TICKS:
                return None
            nearest, _ = _nearest_on_prop(ctx.obs, prop, ctx.here)
            return action.walk(geometry.heading_to(ctx.here, nearest), 0.3, "none")

        if self.started:
            # Our expression shows last tick's result: someone else holding the prop leaves it "none".
            holding = me.expression(ctx.obs)
            self.misses = 0 if holding["type"] == "use" and holding["target"] == self.prop_id else self.misses + 1
            if self.misses >= 3:
                self.blocked = True
                return None
        self.started = True
        self.left -= 1
        look_at = self.face_fn(ctx) if self.face_fn is not None else None
        return ctx.face(look_at or centre, "use")


class Pause(Block):
    """Stand still with an expression for a number of ticks."""

    def __init__(self, ticks: int, expression: str = "none"):
        self.left = ticks
        self.expression = expression

    def tick(self, ctx: Context) -> ThreeBranchesAction | None:
        if self.left <= 0:
            return None
        self.left -= 1
        return ctx.stand(self.expression)


class Sequence(Block):
    """Run child blocks one after another."""

    def __init__(self, *children: Block):
        self.children = list(children)

    def tick(self, ctx: Context) -> ThreeBranchesAction | None:
        while self.children:
            order = self.children[0].tick(ctx)
            if order is not None:
                return order
            self.children.pop(0)
        return None


class Repeat(Block):
    """Rebuild a child from ``factory`` each time it finishes, forever."""

    def __init__(self, factory: Callable[[], Block]):
        self.factory = factory
        self.child = factory()

    def tick(self, ctx: Context) -> ThreeBranchesAction | None:
        # A few rebuilds per tick lets a child that finishes instantly hand over without idling,
        # while a routine with nothing to do cannot spin forever.
        for _ in range(4):
            order = self.child.tick(ctx)
            if order is not None:
                return order
            self.child = self.factory()
        return ctx.stand()


class Interrupt(Block):
    """Run ``routine`` until ``trigger`` offers a higher-priority block, run that block to
    completion, then resume the routine where it left off."""

    def __init__(self, trigger: Callable[[Context], Block | None], routine: Block):
        self.trigger = trigger
        self.routine = routine
        self.active: Block | None = None

    def tick(self, ctx: Context) -> ThreeBranchesAction | None:
        if self.active is None:
            self.active = self.trigger(ctx)
        if self.active is not None:
            order = self.active.tick(ctx)
            if order is not None:
                return order
            self.active = None
        return self.routine.tick(ctx)


class ReplyToHellos(Block):
    """Wrap a role's routine so the villager answers every hello in chat and waves back.

    ``style`` sets how much the reply interrupts the routine: ``"stop"`` turns to face the greeter
    and waves; ``"work"`` keeps working and only waves while walking (a wave mid-use would drop the
    prop); ``"sleep"`` mumbles a reply without waking up.
    """

    def __init__(self, routine: Block, role: str, style: str):
        self.routine = routine
        self.reply = HELLO_REPLIES.get(role, HELLO_REPLIES["wanderer"])
        self.style = style
        self.wave_left = 0
        self.wave_at: str | None = None

    def tick(self, ctx: Context) -> ThreeBranchesAction | None:
        spot_waves(ctx)
        greetings = ctx.greetings
        while greetings.pending:
            greeter = greetings.pending.pop(0)
            greetings.say_hello(greeter, self.reply.format(name=title(greeter)), ctx.tick)
            self.wave_left, self.wave_at = HELLO_WAVE_TICKS, greeter

        if self.style == "sleep" and ctx.memory.get("awake"):
            # A sleeper woken by the visitor greets people like a wanderer from now on.
            self.style, self.reply = "stop", HELLO_REPLIES["sleeper_awake"]
        if self.wave_left <= 0 or self.style == "sleep":
            return self.routine.tick(ctx)
        self.wave_left -= 1

        if self.style == "stop":
            greeter = people.find(ctx.obs, self.wave_at)
            if greeter is not None:
                return ctx.face(greeter["position"], "wave")

        order = self.routine.tick(ctx)
        if order is None or order["action"] == 1:  # 1 is "use": keep working, the chat reply is enough
            return order
        return action.walk(order["heading"], order["speed"], "wave")


# --- Role-specific blocks ----------------------------------------------------------------------


def work_at(prop_id: str, ticks: int, skip_if_busy: bool = False) -> Block:
    """Walk to a prop, then use it for a while."""

    return Sequence(GoTo(lambda ctx: ctx.approach_cells(prop_id)), UseProp(prop_id, ticks, skip_if_busy))


def rotation(steps: list[Callable[[], Block]]) -> Block:
    """Loop through a list of step factories in order, forever."""

    if not steps:
        return Pause(1)
    cursor = {"next": 0}

    def next_step() -> Block:
        step = steps[cursor["next"] % len(steps)]
        cursor["next"] += 1
        return step()

    return Repeat(next_step)


class TakeBreak(Block):
    """After a full work cycle: find a free bench, sit there for a while, then hand back to work.
    A bench seen to be occupied on the way, or held by someone on arrival, is swapped for the next
    nearest. If no bench works out, the worker rests on their feet instead."""

    def __init__(self):
        self.tried: set[str] = set()
        self.bench: str | None = None
        self.walk: GoTo | None = None
        self.sit: UseProp | None = None
        self.rest: Pause | None = None

    def tick(self, ctx: Context) -> ThreeBranchesAction | None:
        order = self._run(ctx)
        ctx.memory["on_break"] = order is not None
        if order is None:
            ctx.memory.pop("break_talker", None)
            ctx.memory.pop("break_lines_said", None)
        return order

    def _run(self, ctx: Context) -> ThreeBranchesAction | None:
        if self.rest is not None:
            return self.rest.tick(ctx)
        while True:
            if self.bench is None:
                self.bench = self._pick_bench(ctx)
                if self.bench is None:
                    self.rest = Pause(BREAK_TICKS // 2)
                    return self.rest.tick(ctx)
                self.tried.add(self.bench)
                self.walk = GoTo(lambda c, bench=self.bench: c.approach_cells(bench))
                self.sit = UseProp(self.bench, BREAK_TICKS, face_fn=_face_talker)

            if self.walk is not None:
                if props.state(ctx.obs, self.bench) == "occupied":
                    self.bench = None
                    continue
                order = self.walk.tick(ctx)
                if order is not None:
                    return order
                if self.walk.failed:
                    self.bench = None
                    continue
                self.walk = None

            order = self.sit.tick(ctx)
            if order is not None:
                return order
            if self.sit.started and not self.sit.blocked:
                return None  # sat out the whole break
            self.bench = None

    def _pick_bench(self, ctx: Context) -> str | None:
        if len(self.tried) >= BREAK_BENCH_TRIES:
            return None
        benches = [
            prop
            for prop in props.all(ctx.obs)
            if prop["type"] == "bench" and prop["id"] not in self.tried and props.state(ctx.obs, prop["id"]) != "occupied"
        ]
        if not benches:
            return None
        nearest = min(benches, key=lambda prop: geometry.distance(ctx.here, _prop_center(ctx.obs, prop)))
        return str(nearest["id"])


def _face_talker(ctx: Context) -> dict[str, float] | None:
    """While seated on a break, turn toward the wanderer who is chatting with us."""

    talker = ctx.memory.get("break_talker")
    person = people.find(ctx.obs, talker) if talker else None
    return None if person is None else person["position"]


def on_break(observation: ThreeBranchesObservation, person) -> bool:
    """Whether a seen worker is on break: sitting on a bench, which nobody uses for work."""

    expression = person.get("expression")
    if not expression or expression["type"] != "use":
        return False
    prop = props.find(observation, expression["target"])
    return prop is not None and prop["type"] == "bench"


class Follow:
    """Close in on a person who may be moving: walk straight at them when the way is clear and
    short, otherwise follow a route to the cells around them, refreshed as they move."""

    def __init__(self):
        self.walker: GoTo | None = None
        self.age = 0

    def step(self, ctx: Context, target) -> ThreeBranchesAction:
        gap = geometry.distance(ctx.here, target)
        if gap < 6.0 and ctx.nav.line_clear(ctx.here, target):
            return action.walk(geometry.heading_to(ctx.here, target), 1.0, "none")
        target_cell = layout.cell_at(ctx.obs, target)
        if target_cell is None:
            return ctx.stand()
        self.age += 1
        if self.walker is None or self.age > 15:
            cell = (target_cell["x"], target_cell["y"])
            self.walker = GoTo(lambda c, cell=cell: _cells_near(c.graph, cell, 2))
            self.age = 0
        order = self.walker.tick(ctx)
        return order if order is not None else ctx.face(target)

    def resume(self, ctx: Context) -> ThreeBranchesAction:
        """Keep walking toward where the person was last seen while they are out of sight."""

        order = self.walker.tick(ctx) if self.walker is not None else None
        return order if order is not None else ctx.stand()


class VisitWorker(Block):
    """Find a worker at one of their work sites, wave hello, then stand and watch them work. A
    worker found on a break gets a chat instead."""

    def __init__(self, worker_id: str, sites: list[str]):
        self.worker_id = worker_id
        self.sites = sites[:2]  # look in at most two places before moving on
        self.phase = "search"
        self.search: GoTo | None = None
        self.follow = Follow()
        self.chat: BreakChat | None = None
        self.lost = 0
        self.left = 0

    def tick(self, ctx: Context) -> ThreeBranchesAction | None:
        if self.chat is not None:
            return self.chat.tick(ctx)

        worker = people.find(ctx.obs, self.worker_id)
        if self.phase == "search":
            if worker is not None:
                self.phase = "approach"
            else:
                return self._search(ctx)

        if worker is None:
            self.lost += 1
            if self.lost > LOST_TICKS:
                return None
            return self.follow.resume(ctx) if self.phase == "approach" else ctx.stand()
        self.lost = 0

        if on_break(ctx.obs, worker) and _break_chat_allowed(ctx, self.worker_id):
            self.chat = BreakChat(self.worker_id)
            return self.chat.tick(ctx)

        gap = geometry.distance(ctx.here, worker["position"])
        if self.phase == "approach":
            if gap <= OBSERVE_DISTANCE:
                self.phase, self.left = "greet", GREET_TICKS
                ctx.greetings.say_hello(self.worker_id, f"Hello, {title(self.worker_id)}! Mind if I watch?", ctx.tick)
            else:
                return self.follow.step(ctx, worker["position"])
        if self.phase == "greet":
            if self.left > 0:
                self.left -= 1
                return ctx.face(worker["position"], "wave")
            self.phase, self.left = "observe", OBSERVE_TICKS
        if self.left <= 0:
            return None
        self.left -= 1
        if gap > OBSERVE_DISTANCE * 2:
            # The worker moved on to their next job: tag along and keep watching.
            return self.follow.step(ctx, worker["position"])
        nod = self.left % 25 == 0
        return ctx.face(worker["position"], "nod" if nod else "none")

    def _search(self, ctx: Context) -> ThreeBranchesAction | None:
        while self.sites:
            if self.search is None:
                site = self.sites[0]
                self.search = GoTo(lambda c, site=site: c.approach_cells(site))
            order = self.search.tick(ctx)
            if order is not None:
                return order
            self.sites.pop(0)
            self.search = None
        return None


class Chat(Block):
    """Walk up to another wanderer and trade emotes with them for a while."""

    def __init__(self, partner_id: str):
        self.partner_id = partner_id
        self.left = CHAT_TICKS
        self.approach = 20
        self.lost = 0
        self.greeted = False

    def tick(self, ctx: Context) -> ThreeBranchesAction | None:
        if not self.greeted:
            self.greeted = True
            fellow = "fellow " if ROLES.get(self.partner_id) == ROLES.get(ctx.me_id) else ""
            greeting = f"Hello, {fellow}{title(self.partner_id)}! Fancy meeting you here."
            ctx.greetings.say_hello(self.partner_id, greeting, ctx.tick)
        partner = people.find(ctx.obs, self.partner_id)
        if partner is None:
            self.lost += 1
            return None if self.lost > 5 else ctx.stand()
        self.lost = 0
        gap = geometry.distance(ctx.here, partner["position"])
        if gap > CHAT_DISTANCE and self.approach > 0:
            self.approach -= 1
            if ctx.nav.line_clear(ctx.here, partner["position"]):
                return action.walk(geometry.heading_to(ctx.here, partner["position"]), 0.8, "wave")
            return ctx.face(partner["position"], "wave")
        if gap > CHAT_SEE_RANGE or self.left <= 0:
            return None
        self.left -= 1
        emote = CHAT_EMOTES[(CHAT_TICKS - self.left) // 6 % len(CHAT_EMOTES)]
        return ctx.face(partner["position"], emote)


class BreakChat(Block):
    """Walk over to a worker sitting out their break and make small talk until it ends. The worker
    stays seated, so they answer in chat while the wanderer does the emoting."""

    def __init__(self, worker_id: str):
        self.worker_id = worker_id
        self.follow = Follow()
        self.phase = "approach"
        self.left = BREAK_CHAT_TICKS
        self.line = 0
        self.lost = 0

    def tick(self, ctx: Context) -> ThreeBranchesAction | None:
        worker = people.find(ctx.obs, self.worker_id)
        if worker is None:
            self.lost += 1
            return None if self.lost > 5 else self.follow.resume(ctx)
        self.lost = 0
        # A seen record shows whether they are still sitting; a heard one only shows where they are.
        if "expression" in worker and not on_break(ctx.obs, worker):
            return None  # the break is over

        gap = geometry.distance(ctx.here, worker["position"])
        if self.phase == "approach":
            if gap > CHAT_DISTANCE + 0.4:
                return self.follow.step(ctx, worker["position"])
            self.phase = "talk"
            ctx.greetings.say_hello(self.worker_id, f"Hello, {title(self.worker_id)}! Taking a break?", ctx.tick)
            ctx.memory.setdefault("break_chats", {})[self.worker_id] = ctx.tick + BREAK_CHAT_COOLDOWN_TICKS
            return ctx.face(worker["position"], "wave")

        if self.left <= 0:
            return None
        self.left -= 1
        talked = BREAK_CHAT_TICKS - self.left
        if talked % BREAK_CHAT_LINE_TICKS == 0 and self.line < len(SMALL_TALK):
            ctx.greetings.say(self.worker_id, SMALL_TALK[self.line])
            self.line += 1
        emote = BREAK_CHAT_EMOTES[talked // 8 % len(BREAK_CHAT_EMOTES)]
        return ctx.face(worker["position"], emote)


def _break_chat_allowed(ctx: Context, worker_id: str) -> bool:
    return ctx.memory.setdefault("break_chats", {}).get(worker_id, 0) <= ctx.tick


def is_wandering(observation: ThreeBranchesObservation, person) -> bool:
    """Whether someone is out wandering: a wanderer, or a sleeper who has been woken up. A sleeper
    only ever sleeps indoors, so one seen out of doors must be awake."""

    role = ROLES.get(str(person["id"]))
    if role == "wanderer":
        return True
    return role == "sleeper" and layout.building_at(observation, person["position"]) is None


def wanderer_social_trigger(ctx: Context) -> Block | None:
    """Interrupt a wanderer's rounds to chat with a nearby wanderer, or with a worker on break."""

    cooldowns: dict[str, int] = ctx.memory.setdefault("chat_cooldowns", {})
    candidates = [
        (person, CHAT_SEE_RANGE) for person in people.seen(ctx.obs)
    ] + [(person, CHAT_HEAR_RANGE) for person in people.nearby(ctx.obs)]
    for person, reach in candidates:
        other = str(person["id"])
        if other == ctx.me_id or not is_wandering(ctx.obs, person):
            continue
        if cooldowns.get(other, 0) > ctx.tick:
            continue
        if geometry.distance(ctx.here, person["position"]) <= reach:
            cooldowns[other] = ctx.tick + CHAT_COOLDOWN_TICKS
            return Chat(other)

    for person in people.seen(ctx.obs):
        worker = str(person["id"])
        if ROLES.get(worker) not in ("blacksmith", "priest", "gardener") or not on_break(ctx.obs, person):
            continue
        if not _break_chat_allowed(ctx, worker):
            continue
        if geometry.distance(ctx.here, person["position"]) <= BREAK_CHAT_RANGE:
            ctx.memory["break_chats"][worker] = ctx.tick + BREAK_CHAT_COOLDOWN_TICKS
            return BreakChat(worker)
    return None


# --- Role trees --------------------------------------------------------------------------------


def build_role(observation: ThreeBranchesObservation, player_id: str) -> Block:
    """Return this villager's whole tree: their role's routine, wrapped so they answer hellos."""

    role = ROLES.get(player_id, "wanderer")
    style = {"wanderer": "stop", "sleeper": "sleep"}.get(role, "work")
    return ReplyToHellos(build_routine(observation, player_id, role), role, style)


def build_routine(observation: ThreeBranchesObservation, player_id: str, role: str) -> Block:
    sites = work_sites(observation, player_id)

    if role in ("blacksmith", "priest", "gardener"):
        repair = set(_props_of(observation, "repair_bench"))
        ticks = {
            "repair_bench": REPAIR_BENCH_WORK_TICKS,
            "stall": STALL_WORK_TICKS,
            "pump": PUMP_TICKS,
            "shrine": SHRINE_TICKS,
            "plot": PLOT_WORK_TICKS,
        }
        steps = [
            lambda site=site: work_at(
                site, ticks.get(str(props.find(observation, site)["type"]), STALL_WORK_TICKS), skip_if_busy=site in repair
            )
            for site in sites
        ]
        # One full lap of the work sites, then a break on a bench, then back to work.
        return rotation(steps + [TakeBreak] if steps else [])

    if role == "sleeper":
        return Sleeper(observation, player_id)
    return wanderer_routine(observation, player_id)


SLEEP_ACTION = action.stand(0.0, "sleep")["action"]


def startled_by_visitor(ctx: Context) -> dict[str, float] | None:
    """Return where the visitor is if we can see them doing the startle emote close by."""

    for person in people.seen(ctx.obs):
        if not people.is_visitor(str(person["id"])) or person["expression"]["type"] != "startle":
            continue
        if geometry.distance(ctx.here, person["position"]) <= geometry.HEARING_RANGE:
            return person["position"]
    return None


class Sleeper(Block):
    """Sleep at home all day, turning over to face a visitor heard nearby. If the visitor startles
    them, they jump awake and spend the rest of the day wandering like a wanderer."""

    def __init__(self, observation: ThreeBranchesObservation, player_id: str):
        home = me.home(observation)
        self.asleep = Sequence(
            GoTo(lambda c: _home_cells(c.obs, c.graph, home) if home != "none" else ()),
            Repeat(lambda: Pause(100, "sleep")),
        )
        self.wake_into = lambda: wanderer_routine(observation, player_id)
        self.awake: Block | None = None
        self.jolt = 0
        self.visitor_at: dict[str, float] | None = None

    def tick(self, ctx: Context) -> ThreeBranchesAction | None:
        if self.awake is None:
            self.visitor_at = startled_by_visitor(ctx)
            if self.visitor_at is not None:
                ctx.memory["awake"] = True
                self.awake, self.jolt = self.wake_into(), WAKE_TICKS
                ctx.greetings.say(VISITOR_ID, WAKE_LINE)

        if self.awake is None:
            order = self.asleep.tick(ctx)
            visitor = people.find(ctx.obs, VISITOR_ID)
            if order is not None and order["action"] == SLEEP_ACTION and visitor is not None:
                # Stir and roll over toward the noise, so a startle right here is seen.
                return ctx.face(visitor["position"], "sleep")
            return order

        if self.jolt > 0:
            self.jolt -= 1
            return ctx.face(self.visitor_at, "startle")
        return self.awake.tick(ctx)


def wanderer_routine(observation: ThreeBranchesObservation, player_id: str) -> Block:
    """Visit each worker in turn, with chats with other wanderers and with workers on a break
    taking priority."""

    roster_ids = {str(entry["id"]) for entry in people.roster(observation)}
    workers = [worker for worker in WORKER_IDS if worker in roster_ids]
    # Each wanderer starts its rounds with a different worker; a woken sleeper takes its own slot.
    offset = WANDERER_IDS.index(player_id) * 2 if player_id in WANDERER_IDS else 1
    workers = workers[offset:] + workers[:offset]
    cursor = {"next": 0, "site": {}}

    def next_visit() -> Block:
        if not workers:
            return Pause(30, "none")
        worker = workers[cursor["next"] % len(workers)]
        cursor["next"] += 1
        worker_sites = work_sites(observation, worker)
        if not worker_sites:
            return Pause(1)
        start = cursor["site"].get(worker, 0)
        cursor["site"][worker] = start + 1
        ordered = worker_sites[start % len(worker_sites) :] + worker_sites[: start % len(worker_sites)]
        return VisitWorker(worker, ordered)

    return Interrupt(wanderer_social_trigger, Repeat(next_visit))


# --- Agent -------------------------------------------------------------------------------------


class Agent:
    """One villager. Its role decides which behaviour tree it runs all day."""

    def reset(self, seed: int, observation: ThreeBranchesObservation) -> None:
        """Build the walk graph and this villager's behaviour tree before the day starts."""

        self._player_id = me.player_id(observation)
        self._nav = Navigator.for_village(observation)
        self._approach: dict[str, frozenset[Cell]] = {}
        self._memory: dict = {}
        self._greetings = Greetings(self._player_id)
        self._tick = 0
        self._tree = build_role(observation, self._player_id)

    def _context(self, observation: ThreeBranchesObservation) -> Context:
        here = me.position(observation)
        found = layout.cell_at(observation, here)
        return Context(
            obs=observation,
            me_id=self._player_id,
            here=here,
            cell=None if found is None else (found["x"], found["y"]),
            tick=day.tick(observation),
            nav=self._nav,
            approach=self._approach,
            greetings=self._greetings,
            memory=self._memory,
        )

    def act(self, observation: ThreeBranchesObservation) -> ThreeBranchesAction:
        """Advance this villager's behaviour tree by one tick."""

        ctx = self._context(observation)
        self._tick = ctx.tick
        # Everyone we can hear can hear us, so these are the villagers we can message directly.
        self._greetings.hearers = frozenset(str(person["id"]) for person in people.nearby(observation))
        order = self._tree.tick(ctx)
        return order if order is not None else ctx.stand()

    def chat(self, inbox: list[dict]) -> list[dict] | None:
        """Note any hellos for the next ``act`` to answer, chat back to wanderers who talk to us on
        a break, and send everything queued so far."""

        for message in inbox or ():
            if not isinstance(message, dict):
                continue
            sender = str(message.get("from"))
            if self._memory.get("on_break") and ROLES.get(sender) in ("wanderer", "sleeper"):
                self._memory["break_talker"] = sender
            if self._greetings.is_hello_for_me(message):
                self._greetings.heard_hello(sender, self._tick)
            elif self._memory.get("on_break") and ROLES.get(sender) in ("wanderer", "sleeper") and message.get("to") == self._player_id:
                self._reply_on_break(sender)
        outgoing, self._greetings.outbox = self._greetings.outbox, []
        return outgoing

    def _reply_on_break(self, wanderer_id: str) -> None:
        """Answer a wanderer's small talk with the next of this role's break lines."""

        lines = BREAK_REPLIES.get(ROLES.get(self._player_id, ""), BREAK_REPLIES["gardener"])
        said = self._memory.get("break_lines_said", 0)
        self._memory["break_lines_said"] = said + 1
        self._greetings.say(wanderer_id, lines[said % len(lines)])
