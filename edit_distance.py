"""Задание 1. Расстояние Левенштейна и Дамерау–Левенштейна (вариант OSA)
с восстановлением последовательности операций.

Рекуррентная формула (D[i][j] — расстояние между a[:i] и b[:j]):
    D[i][j] = min( D[i-1][j] + 1,            # удаление a[i-1]
                   D[i][j-1] + 1,            # вставка b[j-1]
                   D[i-1][j-1] + [a[i-1] != b[j-1]] )   # замена / совпадение
Дамерау добавляет перестановку соседних символов:
    если a[i-1] == b[j-2] и a[i-2] == b[j-1]:
        D[i][j] = min(D[i][j], D[i-2][j-2] + 1)
Сложность: время O(n*m), память O(n*m).
"""


def distance_matrix(a, b, damerau=False):
    n, m = len(a), len(b)
    D = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        D[i][0] = i
    for j in range(m + 1):
        D[0][j] = j

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            D[i][j] = min(D[i - 1][j] + 1,          # удаление
                          D[i][j - 1] + 1,          # вставка
                          D[i - 1][j - 1] + cost)   # замена / совпадение
            if (damerau and i > 1 and j > 1
                    and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]):
                D[i][j] = min(D[i][j], D[i - 2][j - 2] + 1)  # перестановка
    return D


def backtrace(D, a, b, damerau=False):
    """Идём из правого нижнего угла матрицы назад и смотрим,
    из какой клетки получилось значение. Каждая операция: (тип, из_a, в_b)."""
    i, j = len(a), len(b)
    ops = []
    while i > 0 or j > 0:
        if i > 0 and j > 0 and a[i - 1] == b[j - 1] and D[i][j] == D[i - 1][j - 1]:
            ops.append(("match", a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif (damerau and i > 1 and j > 1
              and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]
              and D[i][j] == D[i - 2][j - 2] + 1):
            ops.append(("swap", a[i - 2:i], b[j - 2:j]))
            i, j = i - 2, j - 2
        elif i > 0 and j > 0 and D[i][j] == D[i - 1][j - 1] + 1:
            ops.append(("sub", a[i - 1], b[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and D[i][j] == D[i - 1][j] + 1:
            ops.append(("del", a[i - 1], ""))
            i -= 1
        else:
            ops.append(("ins", "", b[j - 1]))
            j -= 1
    return ops[::-1]


def levenshtein(a, b):
    return distance_matrix(a, b)[-1][-1]


def damerau_levenshtein(a, b):
    return distance_matrix(a, b, damerau=True)[-1][-1]


def edit_ops(a, b, damerau=False):
    """Возвращает (расстояние, список операций a -> b)."""
    D = distance_matrix(a, b, damerau)
    return D[-1][-1], backtrace(D, a, b, damerau)


def show_ops(ops):
    """Человекочитаемый вывод: только реальные правки."""
    return [f"{op}: '{s}' -> '{d}'" for op, s, d in ops if op != "match"]


if __name__ == "__main__":
    for a, b in [("привет", "првиет"), ("kitten", "sitting"), ("қазақ", "казак")]:
        d1, ops1 = edit_ops(a, b)
        d2, ops2 = edit_ops(a, b, damerau=True)
        print(f"{a} -> {b}")
        print(f"  Левенштейн = {d1}: {show_ops(ops1)}")
        print(f"  Дамерау    = {d2}: {show_ops(ops2)}")
