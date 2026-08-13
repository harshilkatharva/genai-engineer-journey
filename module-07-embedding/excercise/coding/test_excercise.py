import pytest

from .excercise import chunk_text_with_overlap, cosine_similarity


def test_cosine_similarity():
    score1 = cosine_similarity([1.0, 1.0, 2.1], [1.0, 1.0, 2.1])
    score2 = cosine_similarity([1.0, 1.0, 2.1], [-1.0, -1.0, -2.1])

    assert score1 == pytest.approx(1)
    assert score2 == pytest.approx(-1)


def test_chunk_overlap():
    chunks = chunk_text_with_overlap(
        "one two three four five six seven eight nine ten eleven", chunk_size=5, overlap=2
    )
    print(chunks)

    for i in range(len(chunks) - 1):
        current = chunks[i].split()
        next_chunk = chunks[i + 1].split()
        print(current)
        assert current[-2:] == next_chunk[:2]
