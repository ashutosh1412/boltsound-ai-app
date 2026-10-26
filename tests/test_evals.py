from evals.run import check


def test_check_passes_a_good_answer():
    case = {"route": "question", "include": ["30"], "exclude": ["14 days"], "source": "returns.md"}
    assert check(case, "question", "Yes, within 30 days. [returns.md]") == []


def test_check_lists_every_problem():
    case = {"route": "question", "include": ["30"], "source": "returns.md"}
    problems = check(case, "small_talk", "Sure! 14 days.")
    assert len(problems) == 3
