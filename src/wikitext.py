import numpy as np
import re
import os
from pathlib import Path

class Wikitext103Generator:
    """
    Parsing code copied from
    https://dax-cdn.cdn.appdomain.cloud/dax-wikitext-103/1.0.1/data-preview/WikiText_103_Notebook.html
    """

    def __init__(self, **kwargs):
        self.datadir = kwargs.get("datadir", os.path.join(os.path.abspath(os.path.dirname(__file__)), "datasets"))
        self.seed = kwargs.get("seed", None)
        if self.seed is None:
            random_data = os.urandom(4)
            self.seed = int.from_bytes(random_data, byteorder="big")

        self.n_resets = 0
        self.rng = np.random.default_rng(self.seed)

        self.load()
        self.reshuffle()

    def load(self):

        # Read train, val, and test sets into string objects
        train_data = Path(os.path.join(self.datadir, 'wikitext-103/wiki.train.tokens')).read_text()
        # val_data = Path(os.path.join(self.datadir, 'wikitext-103/wiki.valid.tokens')).read_text()
        # test_data = Path(os.path.join(self.datadir, 'wikitext-103/wiki.test.tokens')).read_text()

        # Store regular expression pattern to search for wikipedia article headings
        heading_pattern = '( \n \n = [^=]*[^=] = \n \n )'

        # Split out train headings and articles
        self.train_split = re.split(heading_pattern, train_data)
        self.train_headings = [x[7:-7] for x in self.train_split[1::2]]
        self.train_articles = [x for x in self.train_split[2::2]]

        # # Split out validation headings and articles
        # val_split = re.split(heading_pattern, val_data)
        # val_headings = [x[7:-7] for x in val_split[1::2]]
        # val_articles = [x for x in val_split[2::2]]
        #
        # # Split out test headings and articles
        # test_split = re.split(heading_pattern, test_data)
        # test_headings = [x[7:-7] for x in test_split[1::2]]
        # test_articles = [x for x in test_split[2::2]]

    def reshuffle(self):
        self.iter_order = np.arange(len(self))
        self.rng.shuffle(self.iter_order)
        self.iter_ctr = 0
        self.n_resets += 1
        pass

    def __len__(self):
        return len(self.train_headings)

    def __next__(self):
        if self.iter_ctr >= len(self):
            self.reshuffle()
        next_id = self.iter_order[self.iter_ctr]
        dct = {"heading": self.train_headings[next_id],
               "content": self.train_articles[next_id]}
        self.iter_ctr += 1
        return dct


