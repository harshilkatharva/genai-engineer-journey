from rag_app.evalution.report import EvalutionReport


def test_calculate_answer_accuracy_requires_all_required_answers_and_one_alternative():
    report = EvalutionReport.__new__(EvalutionReport)

    result = report._calculate_answer_accuracy(
        {
            "expected_answer": {
                "required": ["Pharaon", "February 24, 1815"],
                "any_of": ["M. Madeleine", "Father Madeleine"],
            }
        },
        "The ship was PHARAON on February 24, 1815. He later used the name Father Madeleine.",
    )

    assert result["accuracy"] == 1.0
    assert result["matched_required_answers"] == ["February 24, 1815", "Pharaon"]
    assert result["matched_any_of_answers"] == ["Father Madeleine"]


def test_calculate_answer_accuracy_reports_partial_required_matches():
    report = EvalutionReport.__new__(EvalutionReport)

    result = report._calculate_answer_accuracy(
        {
            "expected_answer": {
                "required": ["Breaking a pane of glass and stealing a loaf"],
                "any_of": [],
            }
        },
        "He was sentenced after breaking a pane of glass.",
    )

    assert result["accuracy"] == 0.0
    assert result["missing_required_answers"] == ["Breaking a pane of glass and stealing a loaf"]


def test_calculate_answer_accuracy_normalizes_typographic_punctuation():
    report = EvalutionReport.__new__(EvalutionReport)

    result = report._calculate_answer_accuracy(
        {"expected_answer": {"required": ["Château d’If"], "any_of": []}},
        "Edmond was imprisoned in Château d'If.",
    )

    assert result["accuracy"] == 1.0
