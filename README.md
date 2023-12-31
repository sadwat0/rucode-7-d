# RuCode7.0 Final. Расстановка ударений
## 1 место 🥇 (команда ML Reference)

### Данные
- слова, в которых ударный символ обозначен символом "^" справа
- для слов являющихся омографами (т.е. для слов, в которых возможны несколько разных вариантов расстановки ударений) в датасете использован один из возможных вариантов расстановки.

Примеры слов в обучающем датасете:
- взде^ргивается
- измоча^ливавшееся
- обезья^нств
- отсе^иваниях
- поруче^нец
- спрока^зивший
- слова^ми
- ёфика^торам

### Архитектура
Используется Bidirectional LSTM, затем выход проходит через взвешенную сумму, конкатенируется с входными ембедингами. После этого пару полносвязных слоев.

Количество обучаемых параметров модели - ``965 358``.
Время обучения в google colab - примерно 2 часа.

### Метрики
Старая версия получала `97.4%` точность на публичном датасете, немного дообученная (эта версия) получает `97.79%` на приватном датасете.

### Пути к данным
Обучающий датасет лежит по пути ``data/train_stresses_labels.txt``.

Файл со словами, для которых нужно сделать предсказания - ``data/private_test_stresses.txt``.
