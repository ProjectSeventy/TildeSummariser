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