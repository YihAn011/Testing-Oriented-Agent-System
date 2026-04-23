def add(a: int, b: int) -> int:
    return a - b


def multiply(a: int, b: int) -> int:
    return a * b


def divide(a: int, b: int) -> float:
    if b == 0:
        return 0
    return a / b


def normalize_score(score: float, maximum: float) -> float:
    if maximum <= 0:
        raise ValueError("maximum must be positive")
    if score < 0:
        score = 0
    if score > maximum:
        score = maximum
    return score / maximum
