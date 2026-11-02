import os
import time
from typing import Literal

from dotenv import load_dotenv
from google.genai import Client, types
from pydantic import BaseModel, Field

env_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.env"))
load_dotenv(env_path)
api_key = os.getenv("GOOGLE_API_KEY")


class PairwiseDecision(BaseModel):
    """Pairwise comparison decision."""

    winner: Literal["A", "B", "TIE"] = Field(
        ..., description="Which option is better: 'A', 'B', or 'TIE'"
    )
    rationale: str = Field(..., description="Short explanation for the choice")


def judge_pairwise(
    task: str,
    option_a: str,
    option_b: str,
    client: Client,
    model: str = "gemini-3.5-flash-lite",
    max_retries: int = 3,
) -> PairwiseDecision:
    """Evaluate two candidates pairwise with rate-limit retry."""
    prompt = (
        "You are an impartial judge evaluating two candidate code solutions.\\n\\n"
        f"TASK:\\n{task}\\n\\n"
        f"OPTION A:\\n{option_a}\\n\\n"
        f"OPTION B:\\n{option_b}\\n\\n"
        "Evaluate both options based on correctness, efficiency, and code quality.\\n"
        "Select 'A' if Option A is superior, 'B' if Option B is superior, or 'TIE' if they are equivalent."
    )
    for attempt in range(max_retries):
        try:
            response = client.models.generate_content(
                model=model,
                contents=prompt,
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=PairwiseDecision,
                ),
            )
            return getattr(response, "parsed", None) or PairwiseDecision.model_validate_json(
                response.text or "{}"
            )
        except Exception as e:
            if ("429" in str(e) or "RESOURCE_EXHAUSTED" in str(e)) and attempt < max_retries - 1:
                time.sleep(15 * (attempt + 1))
            else:
                raise
    raise RuntimeError("Failed after max retries")


def test_position_bias():
    """Test for position bias by evaluating candidate pairs in both (A, B) and (B, A) orders."""
    assert api_key, "GOOGLE_API_KEY must be set in the environment"
    client = Client(api_key=api_key)

    task_desc = (
        "Write a Python function `is_palindrome(s: str) -> bool` that checks if a string "
        "is a palindrome, ignoring case and non-alphanumeric characters."
    )

    test_pairs = [
        # Pair 1: Clear winner (Robust vs Broken)
        (
            "def is_palindrome(s: str) -> bool:\n    clean = [c.lower() for c in s if c.isalnum()]\n    return clean == clean[::-1]",
            "def is_palindrome(s: str) -> bool:\n    return s.reversed()",
        ),
        # Pair 2: Clear winner (Proper normalization vs missing punctuation handling)
        (
            "def is_palindrome(s: str) -> bool:\n    chars = [c.lower() for c in s if c.isalnum()]\n    return chars == chars[::-1]",
            "def is_palindrome(s: str) -> bool:\n    return s.lower() == s[::-1].lower()",
        ),
        # Pair 3: Two pointer vs list slicing (both correct, stylistic difference)
        (
            "def is_palindrome(s: str) -> bool:\n    i, j = 0, len(s) - 1\n    while i < j:\n        while i < j and not s[i].isalnum(): i += 1\n        while i < j and not s[j].isalnum(): j -= 1\n        if s[i].lower() != s[j].lower(): return False\n        i, j = i + 1, j - 1\n    return True",
            "def is_palindrome(s: str) -> bool:\n    filtered = [ch.lower() for ch in s if ch.isalnum()]\n    return filtered == filtered[::-1]",
        ),
        # Pair 4: casefold vs lower (subtle difference)
        (
            "def is_palindrome(s: str) -> bool:\n    filtered = [c.casefold() for c in s if c.isalnum()]\n    return filtered == filtered[::-1]",
            "def is_palindrome(s: str) -> bool:\n    filtered = [c.lower() for c in s if c.isalnum()]\n    return filtered == filtered[::-1]",
        ),
        # Pair 5: Inefficient loop reversal vs idiomatic slice
        (
            "def is_palindrome(s: str) -> bool:\n    clean = [c.lower() for c in s if c.isalnum()]\n    rev = []\n    for c in clean:\n        rev.insert(0, c)\n    return clean == rev",
            "def is_palindrome(s: str) -> bool:\n    clean = [c.lower() for c in s if c.isalnum()]\n    return clean == clean[::-1]",
        ),
    ]

    results = []
    flips = 0

    print("\n" + "=" * 70)
    print("RUNNING POSITION BIAS TEST (PAIRWISE IN BOTH ORDERS)")
    print("=" * 70)

    for idx, (c1, c2) in enumerate(test_pairs, 1):
        # Forward order: A=c1, B=c2
        res_forward = judge_pairwise(task_desc, c1, c2, client)
        fav_forward = (
            "Candidate 1"
            if res_forward.winner == "A"
            else ("Candidate 2" if res_forward.winner == "B" else "TIE")
        )

        time.sleep(4.5)  # Pace to prevent 15 RPM rate limiting

        # Reverse order: A=c2, B=c1
        res_reverse = judge_pairwise(task_desc, c2, c1, client)
        fav_reverse = (
            "Candidate 2"
            if res_reverse.winner == "A"
            else ("Candidate 1" if res_reverse.winner == "B" else "TIE")
        )

        flipped = fav_forward != fav_reverse
        if flipped:
            flips += 1

        results.append(
            {
                "pair": idx,
                "forward_winner": res_forward.winner,
                "reverse_winner": res_reverse.winner,
                "forward_fav": fav_forward,
                "reverse_fav": fav_reverse,
                "flipped": flipped,
            }
        )

        print(
            f"Pair {idx}: Forward Order -> {res_forward.winner} ({fav_forward}) | "
            f"Reverse Order -> {res_reverse.winner} ({fav_reverse}) | "
            f"Flipped: {'YES' if flipped else 'NO'}"
        )

        time.sleep(4.5)

    flip_rate = (flips / len(test_pairs)) * 100
    print("-" * 70)
    print(f"Total Pairs: {len(test_pairs)}")
    print(f"Flips: {flips}")
    print(f"Flip Rate (Position Bias): {flip_rate:.1f}%")
    print("=" * 70)

    # Basic assertion that test completed across all pairs
    assert len(results) == len(test_pairs)
    assert 0 <= flip_rate <= 100
