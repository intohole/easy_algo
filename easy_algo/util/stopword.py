import os
from easy_algo.util.file_utils import *


class StopwordReader:
    def __init__(self, stopword_files=None):
        if stopword_files is None:
            stopword_files = get_all_files(
                os.path.join(os.path.dirname(__file__), '..', 'data', 'stop')
                         )
        self.stopword_files = stopword_files
        self.stopwords = set()
        self.load_stopwords()

    def load_stopwords(self):
        for stopword_file in self.stopword_files:
            with open(stopword_file, 'r', encoding='utf-8') as file:
                for line in file:
                    self.stopwords.add(line.strip())

    def get_stopwords(self):
        return self.stopwords
