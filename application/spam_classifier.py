# classifier/application/spam_classifier.py
import re
from application.data import load_train_data
from flask import Flask

NOT_SPAM = 0
SPAM = 1


class MetaSingleton(type):
    _instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super(MetaSingleton, cls).__call__(*args, **kwargs)
        return cls._instances[cls]


class Spam(metaclass=MetaSingleton):
    words_freq = {}
    total_words_count = {}
    count_spam = 0
    pA = 0
    pNotA = 0
    replace_char_list = {
        '!': '',
        '"': '',
        "'": '',
        '.': '',
        ',': '',
        '(': '',
        ')': '',
        '[': '',
        ']': '',
        '/': '',
        '{': '',
        '}': '',
        '?': '',
        '*': '',
        '#': '',
        '№': '',
        ':': '',
        ';': '',
        '%': '',
        '_': '',
        '$': '',
        ' a ': ' ',
        ' of ': ' ',
        ' the ': ' ',
        ' an ': ' ',
        ' as ': ' ',
    }

    # обучить данные
    def train(self, train_words):
        for text in train_words:
            # считаем количество спам предложений
            if text[1] == SPAM:
                self.count_spam += 1
            self._calculate_word_frequencies(text[0], text[1])

        self.pA = self.count_spam / len(train_words)
        self.pNotA = 1 - self.pA

    # расчет количества слов в каждом предложении
    def _calculate_word_frequencies(self, body, label):
        word_list = self._clear_text(body).lower().split(' ')

        for word in word_list:
            if word == '':
                continue

            # общий подсчет слов
            if not self.total_words_count.get(word):
                self.total_words_count[word] = 1


            self.total_words_count[word] += 1

            # количество слов в спаме и нет [не спам, спам]
            if not self.words_freq.get(word):
                self.words_freq[word] = [1, 1]

            self.words_freq[word][label] += 1

    # расчет вероятности слова
    def _calculate_P_Bi_A(self, word, label):
        return self.words_freq[word][label] / self.total_words_count[word]

    # расчет вероятности предложения
    def _calculate_P_B_A(self, text, label):
        word_list = text.lower().split(' ')
        pba = 1
        for word in word_list:
            if not self.words_freq.get(word):
                continue
            pba *= self._calculate_P_Bi_A(word, label)

        return pba

    # отчистить текст от знаков
    def _clear_text(self, text):
        # убрать числа
        text = re.sub(r'\d+', '', text)

        for i, j in self.replace_char_list.items():
            text = text.replace(i, j)

        return text

    # классифицировать
    def classify(self, email):
        email = self._clear_text(email)
        spam_rate = self._calculate_P_B_A(email, SPAM) * self.pA / self.pNotA
        not_spam_rate = self._calculate_P_B_A(email, NOT_SPAM) * self.pA / self.pNotA

        return spam_rate > not_spam_rate


train_data = load_train_data()


def train():
    Spam().train(train_data)
