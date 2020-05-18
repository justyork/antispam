# classifier/application/data.py
import re
import os
import pandas as pd

df = pd.read_csv(os.path.dirname(__file__)+"/raw/spam_or_not_spam.csv")
df = df.dropna()

# так как в программе используется обратные значения, меняем их местами 1 - значение спама
df.label = df.label.replace(to_replace={1: 0, 0: 1})


# быстрый осмотр показал, что во многих письмах встречается слово NUMBER, таким образом мы его полностью уберем
def remove_number(text):
    text = text.replace('NUMBER', '')

    # после удаления останутся лишние пробелы их тоже надо убрать
    text = re.sub(r'\s+', ' ', text)
    return text


# применим функцию
df.email = df.email.apply(remove_number)
# в переменную
train_data = df.values.tolist()


def load_train_data():
    return train_data

# TODO: организовать правильное хранение переработанных данных, не в курсе пока как лучше их хранить
