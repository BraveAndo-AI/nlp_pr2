"""Тесты к заданию 1 (pytest) + замер времени против rapidfuzz.

Запуск тестов:   pytest test_edit_distance.py -v
Замер времени:   python test_edit_distance.py
"""
import random
import time

from rapidfuzz.distance import OSA as RF_OSA
from rapidfuzz.distance import Levenshtein as RF_Lev

from edit_distance import damerau_levenshtein, edit_ops, levenshtein

ALPHABET = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяәғқңөұүһі"


def mutate(s, rnd, k):
    """Делает k случайных правок — получаем похожие строки (как опечатки)."""
    s = list(s)
    for _ in range(k):
        op = rnd.choice("sidt")
        p = rnd.randint(0, len(s))
        if op == "i":
            s.insert(p, rnd.choice(ALPHABET))
        elif p < len(s):
            if op == "s":
                s[p] = rnd.choice(ALPHABET)
            elif op == "d":
                del s[p]
            elif op == "t" and p + 1 < len(s):
                s[p], s[p + 1] = s[p + 1], s[p]
    return "".join(s)


def random_pairs(n=1000, seed=0):
    """Половина пар — совсем случайные строки, половина — строка и её «опечатка»."""
    rnd = random.Random(seed)
    pairs = []
    for k in range(n):
        a = "".join(rnd.choices(ALPHABET, k=rnd.randint(0, 15)))
        if k % 2 == 0:
            b = "".join(rnd.choices(ALPHABET, k=rnd.randint(0, 15)))
        else:
            b = mutate(a, rnd, rnd.randint(1, 3))
        pairs.append((a, b))
    return pairs


PAIRS = random_pairs()


def test_known_examples():
    assert levenshtein("kitten", "sitting") == 3
    assert levenshtein("", "абв") == 3
    assert levenshtein("абв", "абв") == 0
    assert levenshtein("ca", "ac") == 2
    assert damerau_levenshtein("ca", "ac") == 1
    assert levenshtein("привет", "првиет") == 2
    assert damerau_levenshtein("привет", "првиет") == 1


def test_levenshtein_vs_rapidfuzz():
    for a, b in PAIRS:
        assert levenshtein(a, b) == RF_Lev.distance(a, b), (a, b)


def test_damerau_vs_rapidfuzz():
    # наш Дамерау = вариант OSA (каждую подстроку правим не более одного раза)
    for a, b in PAIRS:
        assert damerau_levenshtein(a, b) == RF_OSA.distance(a, b), (a, b)


def test_ops_are_correct():
    """Операции должны: (1) превращать a в b, (2) их число = расстояние."""
    for damerau in (False, True):
        for a, b in PAIRS:
            d, ops = edit_ops(a, b, damerau)
            assert "".join(src for _, src, _ in ops) == a
            assert "".join(dst for _, _, dst in ops) == b
            assert sum(op != "match" for op, _, _ in ops) == d


def benchmark():
    import pandas as pd

    funcs = {
        "Левенштейн (наш)": levenshtein,
        "Левенштейн (rapidfuzz)": RF_Lev.distance,
        "Дамерау/OSA (наш)": damerau_levenshtein,
        "Дамерау/OSA (rapidfuzz)": RF_OSA.distance,
    }
    rows = []
    for name, f in funcs.items():
        t = time.perf_counter()
        for a, b in PAIRS:
            f(a, b)
        total = time.perf_counter() - t
        rows.append({"Метод": name, "Время на 1000 пар, мс": round(total * 1000, 2),
                     "мкс на пару": round(total * 1e6 / len(PAIRS), 2)})
    df = pd.DataFrame(rows)
    print(df.to_markdown(index=False))


if __name__ == "__main__":
    benchmark()
