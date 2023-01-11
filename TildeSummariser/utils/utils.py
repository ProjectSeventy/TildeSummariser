import numpy as np

#Rank sentences based on their relevence to a set of keywords, and number of noun phrases
def rank_sentences_for_relevance(sents, noun_phrases, ranked_keywords, ranked_scores):
    
    sent_scores = []
    ret_sents = []
    current_score = 0
    current_sent = sents[0]
    num_NPs = 0
    
    #For each nounphrase, increase the score by the score of each keyword contained within
    for i in range(len(noun_phrases)):
        np = str(noun_phrases[i]).lower()
        sc = 0
        for j in range(0, len(ranked_keywords)):
            kws = ' '.join(ranked_keywords[j])
            sc += (np.count(kws) * ranked_scores[j])
        
        #If the current nounphrase is part of the same sentence, add the score to the current score and increase the number of nounphrases in that sentence
        if noun_phrases[i].sent == current_sent:
            current_score += sc
            num_NPs += 1
        #Otherwise, update the sentence score as the number of nounphrases * the current score. Update the current sentence, score, and numbner of NPs
        else:
            sent_scores.append(num_NPs*current_score)
            ret_sents.append(current_sent)
            current_sent = noun_phrases[i].sent
            current_score = sc
            num_NPs = 1
    #Add the final sentence score
    sent_scores.append(num_NPs*current_score)
    ret_sents.append(current_sent)
    
    #Order the sentences by score
    ranked_sents = [sent for score, sent in sorted(zip(sent_scores, ret_sents), key = lambda score: score[0], reverse = True)]
    
    return ranked_sents

#Create a summary from a list of ranked sentences
def _ranked_sentences_to_summary(sents, ranked_sents, summary_length):
    
    #Reduce the list of ranked sentences to only as many will make up the summary
    ranked_sents = ranked_sents[0:summary_length]
    summary = []
    
    #For each sentence in the text, if they are in the top N, add them to the summary. This puts the summary sentences in appearance order.
    for sent in sents:
        if sent in ranked_sents:
            summary.append(str(sent))
    
    return summary

#Create a summary from a list of ranked sentences, removing redundant sentences
def ranked_sentences_to_summary_with_redundancy_detection(sents, ranked_sents, summary_length, rte):
    
    #Add the highest ranked sentence to the list of necessary sentences
    summary_sents = [ranked_sents[0]]
    i = 1
    #Whilst not enough sentences have been selected and there are more sentences to select:
    while len(summary_sents) < summary_length and i < len(ranked_sents):
        entailed = False
        #Check current sentence against all previous selected sentences for entailment. If so, check the next sentence, else add sentence to the list of summary sentences
        for j in range(0, len(summary_sents)):
            if rte.check_entailment(summary_sents[j], ranked_sents[i]):
                entailed = True
                break
        if not entailed:
            summary_sents.append(ranked_sents[i])
        i += 1
    
    #Create a summary from the selected sentences
    summary = _ranked_sentences_to_summary(sents, summary_sents, summary_length)
    
    return summary