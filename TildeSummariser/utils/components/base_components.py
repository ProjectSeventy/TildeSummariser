from abc import ABC, abstractmethod


class TextSegmenter(ABC):
    
    @abstractmethod
    def get_segments(self, content)
    
class TopicExtractor(ABC):

    @abstractmethod
    def get_topics(self, segment, n_keywords)

class RedundancyDetector(ABC):

    @abstractmethod
    def check_entailment(self, text, hypo)