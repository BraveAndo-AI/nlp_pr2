"""Задание 4. Сравнение токенизаторов для русского и казахского.

Метрики:
  - размер словаря (уникальные токены на обучающей части; для BPE/SP — vocab_size)
  - доля OOV на отложенной выборке (токены теста, которых нет в словаре; для BPE/SP — доля <unk>)
  - среднее число токенов на слово (токены / слова по пробелам)

Запуск: python tokenizers_cmp.py corpus_ru.txt corpus_kk.txt
Нужно: pip install nltk spacy tokenizers sentencepiece pandas tabulate
       python -c "import nltk; nltk.download('punkt_tab')"
"""
import random
import re
import sys

import pandas as pd

VOCAB_SIZE = 8000


def load_lines(path, seed=42):
    lines = [l.strip().lower() for l in open(path, encoding="utf-8") if l.strip()]
    random.Random(seed).shuffle(lines)
    k = int(len(lines) * 0.9)          # 90% обучение / 10% отложенная выборка
    return lines[:k], lines[k:]


# ---------- словные токенизаторы ----------
def word_tokenizers(lang):
    import nltk
    import spacy

    nlp = spacy.blank(lang)  # "ru" или "kk": только правила токенизации, модель не нужна
    return {
        "пробельный": str.split,
        "regex": lambda t: re.findall(r"\w+|[^\w\s]", t),
        # для казахского punkt нет; word_tokenize от языка почти не зависит
        "nltk": lambda t: nltk.word_tokenize(t, language="russian"),
        "spaCy": lambda t: [tok.text for tok in nlp.make_doc(t)],
    }


def eval_word_tokenizer(tok, train, test):
    vocab = {t for line in train for t in tok(line)}
    test_toks = [t for line in test for t in tok(line)]
    n_words = sum(len(line.split()) for line in test)
    oov = sum(t not in vocab for t in test_toks) / len(test_toks)
    return len(vocab), oov, len(test_toks) / n_words


# ---------- подсловные токенизаторы ----------
def train_bpe(train):
    from tokenizers import Tokenizer, models, pre_tokenizers, trainers

    tok = Tokenizer(models.BPE(unk_token="[UNK]"))
    tok.pre_tokenizer = pre_tokenizers.Whitespace()
    trainer = trainers.BpeTrainer(vocab_size=VOCAB_SIZE, special_tokens=["[UNK]"])
    tok.train_from_iterator(train, trainer)
    return tok


def eval_bpe(tok, test):
    toks = [t for line in test for t in tok.encode(line).tokens]
    n_words = sum(len(line.split()) for line in test)
    oov = sum(t == "[UNK]" for t in toks) / len(toks)
    return tok.get_vocab_size(), oov, len(toks) / n_words


def train_sp(train, prefix):
    import sentencepiece as spm

    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(train), model_prefix=prefix,
        vocab_size=VOCAB_SIZE, model_type="unigram",
        character_coverage=1.0, hard_vocab_limit=False)
    return spm.SentencePieceProcessor(model_file=prefix + ".model")


def eval_sp(sp, test):
    ids = [i for line in test for i in sp.encode(line)]
    n_words = sum(len(line.split()) for line in test)
    oov = sum(i == sp.unk_id() for i in ids) / len(ids)
    return sp.get_piece_size(), oov, len(ids) / n_words


def compare(lang, path):
    train, test = load_lines(path)
    rows = []
    for name, tok in word_tokenizers(lang).items():
        rows.append((name, *eval_word_tokenizer(tok, train, test)))

    bpe = train_bpe(train)
    rows.append(("BPE (tokenizers)", *eval_bpe(bpe, test)))
    sp = train_sp(train, f"sp_{lang}")
    rows.append(("SentencePiece", *eval_sp(sp, test)))

    # пример: как режется одно длинное слово
    word = max(test[0].split(), key=len)
    print(f"[{lang}] '{word}':  BPE={bpe.encode(word).tokens}  SP={sp.encode(word, out_type=str)}")

    df = pd.DataFrame(rows, columns=["Токенизатор", "Словарь", "OOV", "Токенов/слово"])
    df.insert(0, "Язык", lang)
    return df


if __name__ == "__main__":
 #   ru = sys.argv[1] if len(sys.argv) > 1 else "corpus_ru.txt"
    kk = sys.argv[2] if len(sys.argv) > 2 else "corpus_kk.txt"
    res = pd.concat([ compare("kk", kk)])
    res["OOV"] = (res["OOV"] * 100).round(2).astype(str) + "%"
    res["Токенов/слово"] = res["Токенов/слово"].round(3)
    print()
    print(res.to_markdown(index=False))
    res.to_csv("tokenizers_table.csv", index=False)
