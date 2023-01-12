from typing import List

import spacy

from tilde.utils import *


class TildeSummariser():
    
    """
    A class implementing the TildeSummariser
    """
    
    def __init__(
        self,
        curve_coefficients: List[float] = None,
        text_seg: TextSegmenter = None,
        topic_ident: TopicExtractor = None,
        rte: RedundancyDetector = None,
    ):
        """
        Initialises the summariser

        Args:
            curve_coefficients: A list of float coefficients defining a polynomial.
                This is the curve used with TildeDynamic to determine the number of
                sentences to take from each segment.
            text_seg: A TextSegmenter to use to segment the text
            topic_ident: A TopicExtractor to use to extract keyword topics
            rte: A RedundancyDetector to use to determine if a sentence implies another
        """

        self.nlp = spacy.load("en_core_web_sm")
        
        if curve_coefficients == None:
            self.curve_coefficients = [
                0.00025,
                -0.0325,
                0.705,
                34.785,
            ]
        else:
            self.curve_coefficients = curve_coefficients
        
        if text_seg == None:
            self.text_seg = C99Segmenter()
        else:
            self.text_seg = text_seg
            
        if topic_ident == None:
            self.topic_ident = FastRAKE()
        else:
            self.topic_ident = topic_ident
        
        if rte == None:
            self.rte = SimpleRTE()
        else:
            self.rte = rte
    
    def summarise(
        self,
        content: List[str],
        n_total: int,
        n_segment: int = 20,
        dynamic_flag: bool = False
    ) -> List[str]:
        """
        A method to summarise a given text

        Args:
            content: A list of strings of the text to summarise. Each string should be a
                unit larger than a sentence. Paragraphs are preferable.
            n_total: An int of the number of sentences to include in the summary
            n_segment: An int of the number of sentences to extract from each segment, when
                using TildeFixed
            dynamic_flag: A boolean indicator of whether to use TildeFixed or TildeDynamic
        Returns:
            A summary of the text, as a list of sentences
        """
        
        content = list(self.nlp.pipe(content))
        content = [doc for doc in content if not (len(doc) == 1 and doc[0].text == '\n')]
        
        segments, segments_sented = self.text_seg.get_segments(content)
        
        all_summary_sentences = []
        len_prev_seg = 0
        len_all_seg = sum([len(seg) for seg in segments_sented])
        
        for i in range(0, len(segments)):
            if dynamic_flag:
                #Calculate required number of sentences
                last_sent = len(segments_sented[i]) + len_prev_seg
                len_prev_seg = last_sent
                perc = ((last_sent - (len(segments_sented[i])/2))/len_all_seg)*100
                n_segment = self._get_segment_length_from_curve(perc)
                
            all_summary_sentences += self._summarise_segment(segments[i], segments_sented[i], n_segment)

        summary_sentences_as_doc = list(self.nlp.pipe(all_summary_sentences))
        summary_sentences_as_doc = spacy.tokens.Doc.from_docs(summary_sentences_as_doc)
        final_summary_sentences = self._summarise_segment(summary_sentences_as_doc, list(summary_sentences_as_doc.sents), n_total)
        
        return final_summary_sentences
    
    def _summarise_segment(self, segment, sent_list, num_sentences):
        """Summarise a given segment of text"""

        ranked_keywords, ranked_scores = self.topic_ident.get_topics(segment)
    
        noun_phrases = list(segment.noun_chunks)
        ranked_sentences = rank_sentences_for_relevance(sent_list, noun_phrases, ranked_keywords, ranked_scores)
        
        summary_sentences = ranked_sentences_to_summary_with_redundancy_detection(sent_list, ranked_sentences, num_sentences, self.rte)
        summary_sentences = [sentence.rstrip() for sentence in summary_sentences]
        return summary_sentences
    
    def _get_segment_length_from_curve(self, midpoint_percentage):
        """Calculate the number of sentences to extract"""

        num_sentences = 0
        for i in range(len(self.curve_coefficients), 0, step=-1):
            num_sentences += pow(midpoint_percentage, i)*self.curve_coefficients[i]
        num_sentences = round(num_sentences)
        return num_sentences