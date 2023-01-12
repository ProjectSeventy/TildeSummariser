import numpy as np


def rank_sentences_for_relevance(sents, noun_phrases, ranked_keywords, ranked_scores):
    """Rank sentences based on their relevence to a set of keywords, and number of noun phrases"""
    
    sent_scores = []
    ret_sents = []
    current_score = 0
    current_sent = sents[0]
    num_NPs = 0
    
    for i in range(len(noun_phrases)):
        #Calculate score based on score of keywords within noun phrase
        np = str(noun_phrases[i]).lower()
        sc = 0
        for j in range(0, len(ranked_keywords)):
            kws = " ".join(ranked_keywords[j])
            sc += (np.count(kws) * ranked_scores[j])
        
        if noun_phrases[i].sent == current_sent:
            current_score += sc
            num_NPs += 1
        else:
            sent_scores.append(num_NPs*current_score)
            ret_sents.append(current_sent)
            current_sent = noun_phrases[i].sent
            current_score = sc
            num_NPs = 1

    sent_scores.append(num_NPs*current_score)
    ret_sents.append(current_sent)
    
    ranked_sents = [sent for score, sent in sorted(zip(sent_scores, ret_sents), key = lambda score: score[0], reverse = True)]
    
    return ranked_sents

def _ranked_sentences_to_summary(sents, ranked_sents, summary_length):
    """Create a summary from a list of ranked sentences"""
    
    ranked_sents = ranked_sents[0:summary_length]
    summary = []
    
    for sent in sents:
        if sent in ranked_sents:
            summary.append(str(sent))
    
    return summary

def ranked_sentences_to_summary_with_redundancy_detection(sents, ranked_sents, summary_length, rte):
    """Create a summary from a list of ranked sentences, removing redundant sentences"""
    
    summary_sents = [ranked_sents[0]]
    i = 1
    while len(summary_sents) < summary_length and i < len(ranked_sents):
        entailed = False
        
        for j in range(0, len(summary_sents)):
            if rte.check_entailment(summary_sents[j], ranked_sents[i]):
                entailed = True
                break
        
        if not entailed:
            summary_sents.append(ranked_sents[i])
        
        i += 1
    
    summary = _ranked_sentences_to_summary(sents, summary_sents, summary_length)
    return summary