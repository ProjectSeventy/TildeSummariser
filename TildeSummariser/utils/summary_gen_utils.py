from TildeSummariser.utils import check_entailment

#Create a summary from a list of ranked sentences
def ranked_sentences_to_summary(sents, ranked_sents, summary_length):
    
    #Reduce the list of ranked sentences to only as many will make up the summary
    ranked_sents = ranked_sents[0:summary_length]
    summary = []
    
    #For each sentence in the text, if they are in the top N, add them to the summary. This puts the summary sentences in appearance order.
    for sent in sents:
        if sent in ranked_sents:
            summary.append(str(sent))
    
    return summary

#Create a summary from a list of ranked sentences, removing redundant sentences
def ranked_sentences_to_summary_with_redundancy_detection(sents, ranked_sents, summary_length):
    
    #Add the highest ranked sentence to the list of necessary sentences
    summary_sents = [ranked_sents[0]]
    i = 1
    #Whilst not enough sentences have been selected and there are more sentences to select:
    while len(summary_sents) < summary_length and i < len(ranked_sents):
        entailed = False
        #Check current sentence against all previous selected sentences for entailment. If so, check the next sentence, else add sentence to the list of summary sentences
        for j in range(0, len(summary_sents)):
            if check_entailment(summary_sents[j], ranked_sents[i]):
                entailed = True
                break
        if not entailed:
            summary_sents.append(ranked_sents[i])
        i += 1
    
    #Create a summary from the selected sentences
    summary = ranked_sentences_to_summary(sents, summary_sents, summary_length)
    
    return summary