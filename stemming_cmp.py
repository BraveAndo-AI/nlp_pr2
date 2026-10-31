"""Задание 5. Стемминг vs лемматизация на 300 словоформах с эталонной леммой.

Методы:
  ru: Snowball (стеммер), pymorphy3, spaCy (ru_core_news_sm), Stanza
  kk: свой суффиксный стеммер, Stanza

Метрики:
  accuracy  — лемматизатор: pred == лемма;  стеммер: stem(слово) == stem(лемма)
              (т.е. словоформа «склеилась» со своей словарной формой)
  under     — доля пар с ОДНОЙ леммой, получивших РАЗНЫЕ результаты (недорезание)
  over      — доля пар с РАЗНЫМИ леммами, получивших ОДИН результат (перерезание)

Запуск: python stemming_cmp.py gold_lemmas.csv
gold_lemmas.csv: колонки word,lemma,lang (lang = ru / kk)
Совет: бери ~60 лемм × 5 форм — тогда under-stemming считается надёжно.
"""
import sys
from itertools import combinations

import pandas as pd

# ---------- Казахский суффиксный стеммер ----------
# Отрезаем окончания словоизменения (не словообразования!), самые длинные — первыми.
KK_SUFFIXES = sorted(set("""
лар лер дар дер тар тер
ның нің дың дің тың тің
ға ге қа ке на не
ны ні ды ді ты ті
да де та те нда нде
дан ден тан тен нан нен
мен бен пен
ым ім ың ің сы сі мыз міз ымыз іміз ңыз ңіз ыңыз іңіз
ған ген қан кен
атын етін йтын йтін
мын мін бын бін пын пін сың сің
ып іп
""".split()), key=len, reverse=True)


def kk_stem(word, min_len=3):
    """Повторяем: пока слово оканчивается на суффикс и основа остаётся ≥ min_len — режем."""
    word = word.lower()
    changed = True
    while changed:
        changed = False
        for suf in KK_SUFFIXES:
            if word.endswith(suf) and len(word) - len(suf) >= min_len:
                word = word[: -len(suf)]
                changed = True
                break
    return word


# ---------- Загрузка методов ----------
def get_methods(lang):
    """Возвращает {имя: (функция, is_stemmer)}. Недоступные методы пропускаются."""
    methods = {}
    if lang == "ru":
        from nltk.stem.snowball import SnowballStemmer
        snow = SnowballStemmer("russian")
        methods["Snowball"] = (snow.stem, True)

        import pymorphy3
        morph = pymorphy3.MorphAnalyzer()
        methods["pymorphy3"] = (lambda w: morph.parse(w)[0].normal_form, False)

        try:
            import spacy
            nlp = spacy.load("ru_core_news_sm")
            methods["spaCy"] = (lambda w: nlp(w)[0].lemma_.lower(), False)
        except OSError:
            print("spaCy: python -m spacy download ru_core_news_sm")
    else:
        methods["Суффиксный (свой)"] = (kk_stem, True)

    try:
        import stanza
        nlp_st = stanza.Pipeline(lang, processors="tokenize,pos,lemma",
                                 tokenize_pretokenized=True, verbose=False)
        methods["Stanza"] = (lambda w: nlp_st([[w]]).sentences[0].words[0].lemma.lower(), False)
    except Exception as e:
        print(f"Stanza ({lang}) недоступна: {e}. Сначала: stanza.download('{lang}')")
    return methods


# ---------- Метрики ----------
def evaluate(func, is_stemmer, words, lemmas):
    preds = [func(w) for w in words]
    targets = [func(l) for l in lemmas] if is_stemmer else lemmas
    acc = sum(p == t for p, t in zip(preds, targets)) / len(words)

    same = diff = under = over = 0
    for i, j in combinations(range(len(words)), 2):
        if lemmas[i] == lemmas[j]:
            same += 1
            under += preds[i] != preds[j]
        else:
            diff += 1
            over += preds[i] == preds[j]
    return preds, targets, acc, under / max(same, 1), over / max(diff, 1)


if __name__ == "__main__":
    gold = pd.read_csv(sys.argv[1] if len(sys.argv) > 1 else "gold_lemmas.csv")
    gold["word"] = gold["word"].str.lower()
    gold["lemma"] = gold["lemma"].str.lower()

    rows, errors = [], []
    for lang, part in gold.groupby("lang"):
        words, lemmas = part["word"].tolist(), part["lemma"].tolist()
        for name, (func, is_stemmer) in get_methods(lang).items():
            preds, targets, acc, under, over = evaluate(func, is_stemmer, words, lemmas)
            rows.append({"Язык": lang, "Метод": name,
                         "Тип": "стемминг" if is_stemmer else "лемматизация",
                         "accuracy": round(acc, 3),
                         "under-stemming": round(under, 3),
                         "over-stemming": round(over, 4)})
            for w, l, p, t in zip(words, lemmas, preds, targets):
                if p != t:
                    errors.append({"lang": lang, "method": name, "word": w,
                                   "gold_lemma": l, "pred": p, "expected": t})

    res = pd.DataFrame(rows)
    print(res.to_markdown(index=False))
    res.to_csv("stemming_table.csv", index=False)
    pd.DataFrame(errors).to_csv("stemming_errors.csv", index=False)
    print("\nОшибки сохранены в stemming_errors.csv — бери оттуда примеры для отчёта")
