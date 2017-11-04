from elo import elo, expected


def test_expected():
    assert round(expected(1613, 1609), 3) == 0.506
    assert round(expected(1613, 1477), 3) == 0.686
    assert round(expected(1613, 1388), 3) == 0.785
    assert round(expected(1613, 1586), 3) == 0.539
    assert round(expected(1613, 1720), 3) == 0.351

    pairs = [
        (0, 0),
        (1, 1),
        (10, 20),
        (123, 456),
        (2400, 2500),
    ]

    for a, b in pairs:
        assert round(expected(a, b) + expected(b, a), 3) == 1.0


def test_elo():
    exp = 0
    exp += expected(1613, 1609)
    exp += expected(1613, 1477)
    exp += expected(1613, 1388)
    exp += expected(1613, 1586)
    exp += expected(1613, 1720)
    score = (0 + 0.5 + 1 + 1 + 0)

    assert round(elo(1613, exp, score, k=32)) == 1601
    assert round(elo(1613, exp, 3, k=32)) == 1617
