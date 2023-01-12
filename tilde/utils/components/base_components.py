from abc import ABC, abstractmethod
from typing import List, Tuple

from spacy.tokens.span import Span
from spacy.tokens.doc import Doc


class TextSegmenter(ABC):

    """
    A base class for TextSegmenters
    """

    @abstractmethod
    def get_segments(
        self,
        content: List[Doc],
    ) -> Tuple[List[Span], List[List[Span]]]:
        """
        Get the optimal segments from a given text

        Args:
            content: A list of spaCy Docs representing the full text broken into
                super-sentential units - preferably paragraphs
        Returns:
            A list of spaCy Spans, one for each segment
            A list of lists of spaCy Spans, each segment broken into sentences
        """

        pass
    
class TopicExtractor(ABC):

    """
    A base class for TopicExtractors
    """
    
    @abstractmethod
    def get_topics(
        self,
        segment: Span,
        n_keywords: int,
    ) -> Tuple[List[List[str]], List[int]]:
        """
        Get a list of keyword topics from a text

        Args:
            segment: A spaCy Span of the segment to extract keyword topics from
            n_keywords: An int of the number of keyword topics to extract
        Returns:
            A list of keyword topics - each a list of strings
            A corresponding list of keyword scores
        """

        pass

class RedundancyDetector(ABC):

    """
    A base class for RedundancyDetectors
    """

    @abstractmethod
    def check_entailment(
        self,
        text: Span,
        hypo: Span,
    ) -> bool:
        """
        Check if a given text entails another

        Args:
            text: A spaCy Span of the entailing text
            hypo: A spaCy Span of the potentially entailed text
        Returns:
            A boolean indicating whether or not entailment is present
        """

        pass