from collections.abc import Sequence


class InvalidInput(Exception):
    def __init__(self, problems: Sequence[str]) -> None:
        super().__init__("\n".join(problems))
        self.problems = tuple(problems)
