from abc import ABC, abstractmethod


class TextSegmenter(ABC):
    
    @abstractmethod
    def get_segments(self, content):
        pass
    
class TopicExtractor(ABC):

    @abstractmethod
    def get_topics(self, segment, n_keywords):
        pass

class RedundancyDetector(ABC):

    @abstractmethod
    def check_entailment(self, text, hypo):
        pass