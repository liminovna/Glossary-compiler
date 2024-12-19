from collections.abc import Iterator
from collections import defaultdict, Counter
# from typing import Literal, Tuple
import re
from string import punctuation

import pandas as pd
import datetime
from config import ALLOWED_EXTENSIONS, SUPPORTED_LANGUAGES, CUSTOM_STOP_WORDS_RU, CUSTOM_STOP_WORDS_EN
import fitz as pymupdf
import nltk

# nltk.download("stopwords")
# nltk.download('punkt_tab')
# nltk.download('wordnet')


def get_current_timestamp() -> str:
    """
    Get current timestamp in the format 202418121558
    :return:
    """
    return datetime.datetime.now().strftime('%Y%m%d%H%M%S')


class InvalidExtension(Exception):
    def __init__(self, ext):
        self.message = (
            f'Extension {ext} is not supported! Please make sure your file has the extension from the list below:\n{ALLOWED_EXTENSIONS}')

    def __str__(self):
        return self.message


class UnsupportedLanguage(Exception):
    def __init__(self, lang):
        self.message = f'Language {lang} is not supported'

    def __str__(self):
        return self.message


class TextProcessorRU:
    """
    Turns text into list of separate sentences, tokens and lemmas
    """
    import razdel
    # import pymorphy2
    # morph = pymorphy2.MorphAnalyzer()

    # from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords

    STOP_WORDS = tuple(stopwords.words('russian')) + CUSTOM_STOP_WORDS_RU

    @staticmethod
    def separate_sents(text: str) -> list:
        """
        :param text: regular string that consists of one or multiple sentences
        :return: list of sentences
        """
        sentences = []

        # предварительно разбиваем текст со страницы по \n
        # for _ in text.split('\n'):
        #     sentences.extend(list(TextProcessorRU.razdel.sentenize(_)))
        # return sentences
        return [s.text for s in TextProcessorRU.razdel.sentenize(text.replace('\n', ' '))]

    @staticmethod
    def tokenize(sent: str) -> list:
        """
        Divide a string into separate tokens. It is assumed that the string is just one sentence.
        :param sent: sentence
        :return: list of words
        """
        return list(TextProcessorRU.razdel.tokenize(sent))

    @staticmethod
    def lemmatize(words: list) -> list:
        """
        For each word in the list get its lemma
        :param words: list of words
        :return: list of lemmas
        """
        return
        # return [TextProcessorRU.morph.parse(w)[0].normal_form for w in words]


class TextProcessorEN:
    """
    Turns text into list of separate sentences, tokens and lemmas
    """
    import razdel

    # from nltk.tokenize import sent_tokenize, word_tokenize
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords

    nltk.download("stopwords")
    # nltk.download('punkt_tab')
    nltk.download('wordnet')

    STOP_WORDS = tuple(stopwords.words('english')) + CUSTOM_STOP_WORDS_EN
    lemmatizer = WordNetLemmatizer()

    def __init__(self, text):
        self.text = text

    @staticmethod
    def separate_sents(text: str) -> list:
        """
        :param text: regular string that consists of one or multiple sentences
        :return: list of sentences
        """
        sentences = []

        # предварительно разбиваем текст со страницы по \n
        # for _ in text.split('\n'):
        # sentences.extend(list(TextProcessorEN.razdel.sentenize(text.replace('\n', ' '))))
        # return sentences
        return [s.text for s in TextProcessorEN.razdel.sentenize(text.replace('\n', ' '))]

    @staticmethod
    def tokenize(sent: str) -> list:
        """
        Divide a string into separate tokens. It is assumed that the string is just one sentence.
        :param sent: sentence
        :return: list of words
        """
        # добавить правило для ' !!!!
        return [t.text for t in TextProcessorEN.razdel.tokenize(sent)]

    @staticmethod
    def lemmatize(word: str) -> str:
        """
        For each word in the list get its lemma
        :param words: list of words
        :return: list of lemmas
        """
        return TextProcessorEN.lemmatizer.lemmatize(word) # nltk
        # return TextProcessorEN.nlp(word).lemma_ # spacy


class GlossaryCounter:

    def __init__(self, pages: Iterator, ngram_types, lang: str):
        self.pages = pages
        self.ngram_types = ngram_types  # not used
        self.lang = lang
        self.processor = self._text_processor_(self.lang)

        # Counter for lemmas and dictionary for lemmas and indexes of sentences they appear in
        self.feature_counter_uni = Counter()  # {'lemma1': 4, 'lemma2': 2]}
        self.contexts_uni = defaultdict(set)  # {'lemma1': [1,5,9,10]}

        # Counter for bigrams and dictionary for bigrams and indexes of sentences they appear in
        self.feature_counter_bi = Counter()  # {'bigram one': 4, 'bigram 2': 1}
        self.contexts_bi = defaultdict(set)  # {'bigram one': [1,5,9,10]}

        # Dictionary with sentences and their indexes
        self.sent_index = defaultdict()  # {'Second sentence.': 1, 'Third sentence.': 2}
        self.sent_index.default_factory = self.sent_index.__len__

    @staticmethod
    def _text_processor_(lang):
        """
        If lang == 'en' then return class TextProcessorEN, else TextProcessorRU
        :param lang: language in which the book is written
        :return:
        """
        counters = {
            "ru": TextProcessorRU,
            "en": TextProcessorEN
        }

        return counters[lang]

    @staticmethod
    def remove_punct(text) -> str:
        return re.sub(f'[{punctuation.replace("'", '')}]', '', text)

    @staticmethod
    def _preprocess_(text) -> str:
        """
        Touch up the text before it undergoes segmentation and lemmatiozation
        :param text:
        :return:
        """
        return text.replace('-\n', '')  # remove line breaks

    def extract_text(self) -> tuple:
        """
        For every page, update dictionaries with word counts and their locations
        """

        while True:
            try:
                current_page = next(self.pages)
                print(current_page)
                current_text = self._preprocess_(current_page.get_text())

                self.update_dicts(current_text)
            except StopIteration:
                break
            # except Exception as e:
            #     print(e)

        return (
            self.sent_index,

            self.feature_counter_uni,
            self.contexts_uni,

            self.feature_counter_bi,
            self.contexts_bi,
        )

    def update_dicts(self, current_text) -> None:
        """
        Update dictionaries defined in __init__
        :param current_text: text from the current page
        :return:
        """

        for sent_num, initial_sent in enumerate(self.processor.separate_sents(current_text)):
            cleaned_sent = self.remove_punct(initial_sent)
            if cleaned_sent != '':
                lemmas = [self.processor.lemmatize(t) for t in self.processor.tokenize(cleaned_sent)]

                for left_word_idx in range(len(lemmas)):
                    right_word_idx = left_word_idx + 1

                    left_w = lemmas[left_word_idx]
                    if left_w not in self.processor.STOP_WORDS:

                        self.contexts_uni[left_w].add(self.sent_index[initial_sent])

                        self.feature_counter_uni.update([left_w])

                        if right_word_idx < len(lemmas):
                            right_w = lemmas[right_word_idx]
                            if right_w not in self.processor.STOP_WORDS:
                                self.contexts_bi[(left_w, right_w)].add(self.sent_index[initial_sent])

                                self.feature_counter_bi.update([(left_w, right_w)])


class GlossaryCompiler:

    @staticmethod
    def _get_extension_(filename: str) -> str:
        """
        Get extension of the file from the filename
        :param filename:
        :return:
        """
        res = filename.split('.')[-1]
        if res in ALLOWED_EXTENSIONS:
            return res
        raise InvalidExtension

    @staticmethod
    def _check_language_(l: str):
        """
        Check if provided language is supported by the program
        :param l:
        :return:
        """
        if l.lower() in SUPPORTED_LANGUAGES:
            return l.lower()
        raise UnsupportedLanguage(l)

    def __init__(self, filename: str, lang):
        self.filename = filename
        self.extension = self._get_extension_(filename.lower())
        self.lang = self._check_language_(lang)

    def __repr__(self) -> str:
        return f'ready to process {self.extension.upper()}-file {self.filename}'

    @staticmethod
    def _iterate_file_(filename):
        """
        Read txt, pdf, epub, mobi, docx, or fb2 file using pymupdf
        """
        for page in pymupdf.open(filename):
            yield page

    def _build_vocab_(self, ngram_types = (1, 2)):
        """
        Reading into memory one page at a time, divide text into sentences and lists of lemmas and update dictionaries with counts and sentences
        :param ngram_types: 1 - collect statistics on unigrams, 2 - on bigrams. NOT IN USE (but will be in one of the future versions)
        :return:
        """
        if self.extension in ('pdf', 'txt', 'epub', 'docx', 'fb2'):
            pages = self._iterate_file_(self.filename)
        else:
            raise InvalidExtension('This extension is yet to be implemented')

        self.sent_index, self.feature_counter_uni, self.contexts_uni, self.feature_counter_bi, self.contexts_bi = GlossaryCounter(
            pages=pages, ngram_types=ngram_types, lang=self.lang).extract_text()

    def get_top(self, top_n, united=False, min_threshold=0):
        """
        Get only n most frequent entries
        :param top_n: number of mono/bigrams to be exported
        :param united: treat the two dictionaries of uni- and bigrams as one
        :param min_threshold: NOT IN USE
        :return:
        """
        self._build_vocab_()
        df_loc = pd.DataFrame(data=self.sent_index.items(), columns=['sentence', 'sent_idx'])
        if not united:

            top_uni = self._form_dataframe_(self.feature_counter_uni.most_common(top_n), self.contexts_uni, df_loc)
            top_bi = self._form_dataframe_(self.feature_counter_bi.most_common(top_n), self.contexts_bi, df_loc)

            top_bi[['left_word', 'right_word']] = top_bi['lemma'].apply(pd.Series)
            top_bi['bigram'] = top_bi['lemma'].apply(self.join_tuple)
            top_bi = top_bi[['bigram', 'left_word', 'right_word', 'freq', 'sentence']]
            self.output = (top_uni, top_bi)
        else:
            self.output = self._form_dataframe_(
                (self.feature_counter_uni + self.feature_counter_uni).most_common(top_n), self.contexts_bi, df_loc, top_n)

        return self.output

    def _form_dataframe_(self, counts_list, contexts_dict, df_loc):
        df_top = pd.DataFrame(data=counts_list, columns=['lemma', 'freq'])

        df_contxt = pd.DataFrame(data=contexts_dict.items(), columns=['lemma', 'sent_idx'])
        df_contxt = df_contxt.explode('sent_idx')

        df = pd.merge(left=df_top, right=df_contxt, on='lemma', how='left')

        df = pd.merge(left=df, right=df_loc, on='sent_idx', how='left')

        return df

    @staticmethod
    def join_tuple(tpl):
        return ' '.join(tpl)

    def export(self):
        if len(self.output) > 1:
            suffixes = ['unigrams', 'bigrams']
        else:
            suffixes = ['']
        for suffix, table in zip(suffixes, self.output):
            table.to_csv(f'{suffix}_glossary_{get_current_timestamp()}.csv', index=False)

