from __future__ import annotations

from dataclasses import dataclass, replace
from enum import Enum


class Direction(str, Enum):
    UP = "up"
    DOWN = "down"
    LEFT = "left"
    RIGHT = "right"


@dataclass(frozen=True)
class Point:
    row: int
    col: int


@dataclass(frozen=True)
class GameState:
    width: int
    height: int
    snake: tuple[Point, ...]
    direction: Direction
    food: Point
    score: int = 0
    game_over: bool = False

    @property
    def head(self) -> Point:
        return self.snake[0]


DELTAS: dict[Direction, Point] = {
    Direction.UP: Point(-1, 0),
    Direction.DOWN: Point(1, 0),
    Direction.LEFT: Point(0, -1),
    Direction.RIGHT: Point(0, 1),
}


def create_game(width: int = 12, height: int = 8) -> GameState:
    head = Point(height // 2, width // 2)
    snake = (
        head,
        Point(head.row, head.col - 1),
        Point(head.row, head.col - 2),
    )
    food = Point(1, width - 2)
    return GameState(
        width=width,
        height=height,
        snake=snake,
        direction=Direction.RIGHT,
        food=food,
    )


def turn(state: GameState, direction: Direction) -> GameState:
    opposite = {
        Direction.UP: Direction.DOWN,
        Direction.DOWN: Direction.UP,
        Direction.LEFT: Direction.RIGHT,
        Direction.RIGHT: Direction.LEFT,
    }
    if len(state.snake) > 1 and direction == opposite[state.direction]:
        return state
    return replace(state, direction=direction)


def next_head(state: GameState) -> Point:
    delta = DELTAS[state.direction]
    return Point(state.head.row + delta.row, state.head.col + delta.col)


def hits_wall(state: GameState, point: Point) -> bool:
    return point.row < 0 or point.row > state.height or point.col < 0 or point.col > state.width


def step_game(state: GameState, direction: Direction | None = None) -> GameState:
    if state.game_over:
        return state

    if direction is not None:
        state = turn(state, direction)

    new_head = next_head(state)
    ate_food = new_head == state.food

    body_to_check = state.snake if ate_food else state.snake[:-1]
    if hits_wall(state, new_head) or new_head in body_to_check:
        return replace(state, game_over=True)

    new_snake = (new_head, *state.snake)
    new_snake = new_snake[:-1]

    new_food = state.food
    if ate_food:
        new_food = Point(1, 1)

    return replace(
        state,
        snake=new_snake,
        food=new_food,
        score=state.score + (1 if ate_food else 0),
    )


def render(state: GameState) -> str:
    cells = [[" " for _ in range(state.width)] for _ in range(state.height)]
    for part in state.snake[1:]:
        if 0 <= part.row < state.height and 0 <= part.col < state.width:
            cells[part.row][part.col] = "o"
    if 0 <= state.head.row < state.height and 0 <= state.head.col < state.width:
        cells[state.head.row][state.head.col] = "@"
    if 0 <= state.food.row < state.height and 0 <= state.food.col < state.width:
        cells[state.food.row][state.food.col] = "*"

    top_bottom = "+" + "-" * state.width + "+"
    lines = [top_bottom]
    for row in cells:
        lines.append("|" + "".join(row) + "|")
    lines.append(top_bottom)
    lines.append(f"score: {state.score}")
    if state.game_over:
        lines.append("game over")
    return "\n".join(lines)
