from __future__ import annotations

from terminal_snake.game import Direction, create_game, render, step_game


COMMANDS = {
    "w": Direction.UP,
    "a": Direction.LEFT,
    "s": Direction.DOWN,
    "d": Direction.RIGHT,
}


def main() -> None:
    state = create_game()
    print("Terminal Snake")
    print("Use w/a/s/d then press Enter. Press q to quit.")

    while True:
        print()
        print(render(state))
        if state.game_over:
            break
        raw = input("move> ").strip().lower()
        if raw == "q":
            print("bye")
            break
        direction = COMMANDS.get(raw)
        if direction is None:
            print("invalid input, use w/a/s/d or q")
            continue
        state = step_game(state, direction)


if __name__ == "__main__":
    main()
