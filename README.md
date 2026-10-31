# ПР 2. Расстояние редактирования и нормализация текста

## Файлы

| Файл | Задание |
|---|---|
| `edit_distance.py` | 1 — Левенштейн и Дамерау–Левенштейн (OSA) + восстановление операций |
| `test_edit_distance.py` | 1 — тесты (pytest) на 1000 случайных пар + сравнение скорости с rapidfuzz |
| `spell.py` | 2, 3 — корректор Норвига, клавиатурная матрица ошибок, биграммы |
| `tokenizers_cmp.py` | 4 — 6 токенизаторов: словарь, OOV, токенов/слово |
| `stemming_cmp.py` | 5 — Snowball, свой казахский стеммер, pymorphy3, spaCy, Stanza |
| `typos_example.csv`, `gold_lemmas_example.csv` 

## Установка

```bash
pip install rapidfuzz pytest pandas tabulate nltk spacy tokenizers sentencepiece pymorphy3 stanza
python -m spacy download ru_core_news_sm
python -c "import nltk; nltk.download('punkt_tab')"
python -c "import stanza; stanza.download('ru'); stanza.download('kk')"
```

## Запуск

```bash
pytest test_edit_distance.py -v                  # 1: корректность
python test_edit_distance.py                     # 1: таблица времени
python spell.py corpus.txt typos.csv             # 2-3: таблица accuracy@1/@3
python tokenizers_cmp.py corpus_ru.txt corpus_kk.txt   # 4
python stemming_cmp.py gold_lemmas.csv           # 5
```

## Коротко о теории

**Левенштейн** — динамическое программирование:

$$D[i][j] = \min\big(D[i-1][j]+1,\; D[i][j-1]+1,\; D[i-1][j-1] + [a_i \ne b_j]\big)$$

**Дамерау (OSA)** добавляет перестановку: если $a_i=b_{j-1}$ и $a_{i-1}=b_j$, то $D[i][j]=\min(D[i][j],\,D[i-2][j-2]+1)$.
Операции восстанавливаются обратным проходом из $D[n][m]$: смотрим, из какой клетки пришёл минимум.
Пример: `привет → првиет`: Левенштейн = 2 (две замены), Дамерау = 1 (одна перестановка `ив→ви`).

> rapidfuzz: `DamerauLevenshtein` — «полный» вариант, `OSA` — ограниченный. Наш алгоритм = OSA, поэтому сравниваем с `rapidfuzz.distance.OSA`.

**Корректор (noisy channel):**

$$\hat w = \arg\max_w \; P(w)\cdot P(x\mid w)$$

- $P(w)$ — частота слова в корпусе; с контекстом: $P(w\mid prev) = \mu\frac{c(prev,w)}{c(prev)} + (1-\mu)P(w)$
- $P(x\mid w)$ — база: $\lambda^{d}$; улучшение: произведение вероятностей правок (соседняя клавиша 0.05, `қ→к` и др. 0.2, остальные замены 0.005, вставка/удаление 0.01, перестановка 0.03).

**Стеммер для казахского:** казахский — агглютинативный, окончания идут цепочкой (`бала+лар+ға`), поэтому в цикле отрезаем самый длинный суффикс из списка, пока основа ≥ 3 букв.



**1. Сложность Левенштейна и как её снизить.**
Время $O(nm)$, память $O(nm)$ (если нужны операции).
- Память $O(\min(n,m))$: храним только две строки матрицы (если операции не нужны). С операциями — алгоритм Хиршберга, память линейная.
- **Полоса Укконена**: если нужно только $d \le k$, считаем диагональную полосу шириной $2k+1$ → $O(k\cdot\min(n,m))$. Плюс ранний выход, если весь ряд > k.
- **Фильтр по длине**: если $|n-m| > k$, то $d > k$ — пару можно не считать.
- **Битовый параллелизм (Майерс)**: $O(\lceil m/w\rceil \cdot n)$, $w$ — размер машинного слова (так работает rapidfuzz).
- Для поиска по словарю: **BK-дерево**, **SymSpell** (заранее генерируем удаления), префиксное дерево с DP.

**2. Почему BPE смягчает проблему OOV.**
BPE начинает со словаря символов и сливает самые частые пары. Частые слова становятся одним токеном, редкие — собираются из кусков. Любое новое слово можно разбить на известные подслова (в худшем случае — на буквы), поэтому OOV почти нет. Для казахского это особенно важно: `балаларымызға` ≈ `бала + лар + ымыз + ға`, корень и суффиксы — уже знакомые токены. Цена — больше токенов на слово.

**3. Когда стемминг лучше лемматизации.**
- Информационный поиск и индексация — важна полнота и скорость, «некрасивая» основа не мешает.
- Классификация, TF-IDF, тематическое моделирование — нужно лишь склеить формы.
- Большие объёмы текста — стемминг в разы быстрее.
- Малоресурсные языки (казахский), где нет хорошего лемматизатора/словаря.

Лемматизация лучше, когда нужна настоящая словарная форма: генерация текста, машинный перевод, словари, лингвистический анализ, когда важна точность (`лучше → хороший`).

---

## Полезные ссылки
- Jurafsky & Martin, SLP3, гл. 2 (токенизация, BPE, расстояние редактирования): https://web.stanford.edu/~jurafsky/slp3/2.pdf
- Jurafsky & Martin, приложение B (noisy channel, исправление опечаток): https://web.stanford.edu/~jurafsky/slp3/B.pdf
- P. Norvig, How to Write a Spelling Corrector: https://norvig.com/spell-correct.html
- Damerau–Levenshtein (OSA vs полный вариант): https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance
- RapidFuzz: https://rapidfuzz.github.io/RapidFuzz/
- Sennrich et al., BPE для NMT: https://arxiv.org/abs/1508.07909
- HF tokenizers: https://huggingface.co/docs/tokenizers
- SentencePiece: https://github.com/google/sentencepiece
- pymorphy3: https://github.com/no-plagiarism/pymorphy3
- Stanza (есть модели ru и kk): https://stanfordnlp.github.io/stanza/
- SymSpell (быстрый поиск кандидатов): https://github.com/wolfgarbe/SymSpell
