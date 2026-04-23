from terminal_snake.game import Direction, GameState, Point, create_game, render, step_game


def test_render_includes_score_line() -> None:
    state = create_game()
    output = render(state)
    assert "score: 0" in output


def test_eating_food_increases_score_and_grows_snake() -> None:
    state = GameState(
        width=7,
        height=5,
        snake=(Point(2, 2), Point(2, 1), Point(2, 0)),
        direction=Direction.RIGHT,
        food=Point(2, 3),
    )

    updated = step_game(state)

    assert updated.score == 1
    assert updated.head == Point(2, 3)
    assert len(updated.snake) == 4


def test_hitting_right_wall_ends_game() -> None:
    state = GameState(
        width=5,
        height=5,
        snake=(Point(2, 4), Point(2, 3), Point(2, 2)),
        direction=Direction.RIGHT,
        food=Point(0, 0),
    )

    updated = step_game(state)

    assert updated.game_over is True
