"""Задания 2–3. Корректор опечаток по модели Норвига (noisy channel).

    w* = argmax_w  P(w) * P(x | w)
    x — слово с опечаткой, w — кандидат из словаря.

База:      P(w) — частота в корпусе, P(x|w) = λ^d (d — число правок, d ≤ 2).
Улучшение: 1) P(x|w) через взвешенные правки (соседние клавиши ЙЦУКЕН/каз.,
              замена каз. буквы на русскую: қ->к, ә->а …);
           2) контекст: P(w) -> P(w | prev) по биграммам (интерполяция).

Запуск: python spell.py corpus.txt typos.csv
typos.csv: колонки typo,correct,prev  (prev — предыдущее слово, может быть пустым)
"""
import math
import re
import sys
from collections import Counter

from edit_distance import edit_ops

LETTERS = "абвгдеёжзийклмнопрстуфхцчшщъыьэюяәғқңөұүһі"
WORD_RE = re.compile(f"[{LETTERS}]+")

# ---------- Клавиатура: ЙЦУКЕН + казахские буквы в цифровом ряду ----------
KEY_ROWS = [
    " әіңғ  үұқөһ",   # казахская раскладка: каз. буквы на клавишах 2-5, 8-0, -, =
    "йцукенгшщзхъ",
    "фывапролджэ",
    "ячсмитьбю",
]
POS = {ch: (r, c) for r, row in enumerate(KEY_ROWS) for c, ch in enumerate(row) if ch != " "}


def is_neighbor(a, b):
    """Соседние клавиши: отличаются не больше чем на 1 ряд и 1 столбец."""
    if a not in POS or b not in POS:
        return False
    (r1, c1), (r2, c2) = POS[a], POS[b]
    return a != b and abs(r1 - r2) <= 1 and abs(c1 - c2) <= 1


# Очень частая «ошибка»: казахскую букву пишут русской (нет каз. раскладки)
KZ_TO_RU = {"ә": "а", "і": "и", "ң": "н", "ғ": "г", "қ": "к",
            "ө": "о", "ұ": "у", "ү": "у", "һ": "х", "ё": "е"}


def op_prob(op, correct_ch, typed_ch):
    """Вероятность одной правки (правильное слово -> опечатка)."""
    if op == "sub":
        if KZ_TO_RU.get(correct_ch) == typed_ch:
            return 0.2
        if is_neighbor(correct_ch, typed_ch):
            return 0.05
        return 0.005
    if op == "swap":
        return 0.03
    return 0.01  # ins / del


def tokenize(text):
    return WORD_RE.findall(text.lower())


class SpellCorrector:
    def __init__(self, text, lam=0.01, mu=0.7):
        toks = tokenize(text)
        self.words = Counter(toks)
        self.bigrams = Counter(zip(toks, toks[1:]))
        self.N = sum(self.words.values())
        self.lam = lam   # штраф за одну правку в базовой модели
        self.mu = mu     # вес биграммы в интерполяции
        self._cache = {}

    # ---------- кандидаты (как у Норвига) ----------
    @staticmethod
    def edits1(w):
        splits = [(w[:i], w[i:]) for i in range(len(w) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        swaps = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in LETTERS]
        inserts = [L + c + R for L, R in splits for c in LETTERS]
        return set(deletes + swaps + replaces + inserts)

    def candidates(self, x):
        """{слово из словаря: расстояние (0, 1 или 2)}"""
        if x in self._cache:
            return self._cache[x]
        cands = {x: 0} if x in self.words else {}
        e1 = self.edits1(x)
        for w in e1:
            if w in self.words:
                cands.setdefault(w, 1)
        for w1 in e1:
            for w2 in self.edits1(w1):
                if w2 in self.words:
                    cands.setdefault(w2, 2)
        self._cache[x] = cands
        return cands

    # ---------- модель языка P(w) / P(w|prev) ----------
    def p_word(self, w, prev=None, context=False):
        p = self.words[w] / self.N
        if context and prev and self.words[prev]:
            p_bi = self.bigrams[(prev, w)] / self.words[prev]
            p = self.mu * p_bi + (1 - self.mu) * p
        return p

    # ---------- модель ошибок P(x|w) ----------
    def p_typo(self, x, w, d, keyboard=False):
        if d == 0:
            return 0.95
        if not keyboard:
            return self.lam ** d
        _, ops = edit_ops(w, x, damerau=True)
        p = 1.0
        for op, src, dst in ops:
            if op != "match":
                p *= op_prob(op, src, dst)
        return p

    def correct(self, x, prev=None, k=3, keyboard=False, context=False):
        x = x.lower()
        prev = prev.lower() if isinstance(prev, str) and prev else None
        cands = self.candidates(x)
        if not cands:
            return [x]

        def score(w):
            return (math.log(self.p_word(w, prev, context))
                    + math.log(self.p_typo(x, w, cands[w], keyboard)))

        return sorted(cands, key=score, reverse=True)[:k]


def evaluate(sc, df, keyboard=False, context=False):
    top1 = top3 = 0
    errors = []
    for typo, gold, prev in df[["typo", "correct", "prev"]].itertuples(index=False):
        preds = sc.correct(typo, prev, k=3, keyboard=keyboard, context=context)
        gold = gold.lower()
        top1 += preds[0] == gold
        top3 += gold in preds
        if preds[0] != gold:
            errors.append({"prev": prev, "typo": typo, "correct": gold, "top3": ", ".join(preds)})
    n = len(df)
    return top1 / n, top3 / n, errors


if __name__ == "__main__":
    import pandas as pd

    corpus_path = sys.argv[1] if len(sys.argv) > 1 else "corpus.txt"
    typos_path = sys.argv[2] if len(sys.argv) > 2 else "typos.csv"

    sc = SpellCorrector(open(corpus_path, encoding="utf-8").read())
    df = pd.read_csv(typos_path).fillna("")
    print(f"Словарь: {len(sc.words)} слов, тестовых опечаток: {len(df)}\n")

    variants = [("Норвиг (база)", False, False),
                ("+ клавиатура", True, False),
                ("+ биграммы", False, True),
                ("+ клавиатура + биграммы", True, True)]
    rows = []
    for name, kb, ctx in variants:
        a1, a3, errors = evaluate(sc, df, kb, ctx)
        rows.append({"Модель": name, "accuracy@1": round(a1, 3), "accuracy@3": round(a3, 3)})
    res = pd.DataFrame(rows)
    res["прирост @1"] = (res["accuracy@1"] - res.loc[0, "accuracy@1"]).round(3)
    print(res.to_markdown(index=False))

    # ошибки лучшей модели — для разбора в отчёте
    pd.DataFrame(errors).to_csv("spell_errors.csv", index=False)
    print("\nОшибки последней модели сохранены в spell_errors.csv")
