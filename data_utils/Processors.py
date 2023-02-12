import os
import json, csv
from abc import ABC, abstractmethod
from collections import defaultdict, Counter
from typing import List, Dict, Callable

from openprompt.utils.logging import logger

from openprompt.data_utils.utils import InputExample
from openprompt.data_utils.data_processor import DataProcessor

class AgnewsTitleProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = ["World", "Sports", "Business", "Tech"]

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.csv".format(split))
        examples = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, headline = row
                text_a = headline.replace('\\', ' ')
                example = InputExample(guid=str(idx), text_a=text_a, label=int(label)-1)
                examples.append(example)
        return examples
        
        
class SnippetsProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = ["business", "computers", "culture-arts-entertainment", "education-science", "engineering", "health", "politics-society", "sports"]

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.csv".format(split))
        examples = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, headline = row
                text_a = headline.replace('\\', ' ')
                example = InputExample(guid=str(idx), text_a=text_a,label=int(label)-1)
                examples.append(example)
        return examples
        
        

class NewstitleProcessor(DataProcessor):
    def __init__(self):
        super().__init__()
        self.labels = ["business", "entertainment", "health", "technology", "sport", "us", "world"]

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.csv".format(split))
        examples = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, headline = row
                text_a = headline.replace('\\', ' ')
                example = InputExample(guid=str(idx), text_a=text_a,label=int(label)-1)
                examples.append(example)
        return examples
        
        
class CustomProcessor(DataProcessor):
    def __init__(self,labels):
        super().__init__()
        self.labels = labels

    def get_examples(self, data_dir, split):
        path = os.path.join(data_dir, "{}.csv".format(split))
        examples = []
        with open(path, encoding='utf8') as f:
            reader = csv.reader(f, delimiter=',')
            for idx, row in enumerate(reader):
                label, headline = row
                text_a = headline.replace('\\', ' ')
                example = InputExample(guid=str(idx), text_a=text_a,label=int(label)-1)
                examples.append(example)
        return examples
