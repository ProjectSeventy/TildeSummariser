from TildeSummariser.utils.text_seg_utils import get_segments
from TildeSummariser.utils.topic_ident_utils import get_topics
from TildeSummariser.utils.relevance_detect_utils import rank_sentences_for_relevance
from TildeSummariser.utils.redundancy_detect_utils import check_entailment
from TildeSummariser.utils.summary_gen_utils import ranked_sentences_to_summary_with_redundancy_detection

#Get a list of n summary sentences from a span and it's corresponding list of sentences
def get_summary_sentences(segment, sent_list, n):
    ranked_keywords, ranked_scores = get_topics(segment)
    
    noun_phrases = list(segment.noun_chunks)
    ranked_sentences = rank_sentences_for_relevance(sent_list, noun_phrases, ranked_keywords, ranked_scores)
    
    summary_sentences = ranked_sentences_to_summary_with_redundancy_detection(sent_list, ranked_sentences, n)
    return summary_sentences