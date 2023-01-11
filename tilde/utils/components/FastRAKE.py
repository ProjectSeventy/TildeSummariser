from tilde.utils.components.base_components import TopicExtractor


class FastRAKE(TopicExtractor):
    
    def __init__(self):
        self.simple_whitelist = ["NOUN", "PROPN", "ADJ"] 
        self.fine_grain_whitelist = ["VERB:VBG"] 

    #Get a list of keywords form a text
    def get_topics(self, segment, n_keywords=10):
        candidates, appearances, member_words = self._identify_candidates(segment)
        candidates_no_duplication, keyword_scores, candidate_appearances = self._score_all(member_words, candidates, appearances)
        ranked_keywords, ranked_scores = self._prune_candidates(candidates_no_duplication, keyword_scores, candidate_appearances)
        ranked_keywords, ranked_scores = self._trim_keyword_list(ranked_keywords, ranked_scores, n_keywords)
        return ranked_keywords, ranked_scores

    #Identify all candidate keywords in a text, their member words, and their appearances
    def _identify_candidates(self, doc):
        
        candidates = [] #candidate keywords
        member_words = [] #words that make up the above
        appearances = [] #stores indices at which each candidate keyword appears
        
        current_cand = [] #used to build a potential keyword a token at a time
        
        for i, tok in enumerate(doc):
            
            tok_text = tok.text.lower()
            #If a token is of the correct part of speech, add it to current_cand and if the token is not yet in member_words, add it
            if (tok.pos_ in self.simple_whitelist or f"{tok.pos_}:{tok.tag_}" in self.fine_grain_whitelist) and not (tok.is_punct or tok.is_stop):
                current_cand.append(tok_text)
                if tok_text not in member_words:
                    member_words.append(tok_text)
            
            #If token is not of correct part of speech, add the current candidate to the list
            elif len(current_cand) > 0:
                candidates.append(current_cand[:])
                appearances.append(i - len(current_cand))
                current_cand.clear()
        
        #When the document has been iterated through, if there is an un-added candidate, add it
        if len(current_cand) > 0:
            candidates.append(current_cand[:])
            appearances.append(i - len(current_cand))
            current_cand.clear()
        
        return candidates, appearances, member_words

    #Construct a word co-occurrence matrix for all memberWords
    def _make_word_matrix(self, candidates, member_words):
        
        #Construct an empty matrix representing each member word occuring with itself
        word_matrix = []
        for i in range(0, len(member_words)):
            word_row = [0] * len(member_words)
            word_matrix.append(word_row)

        #For each candidate, compare each pairs of words and fill in the relevant matrix cell
        for cand in candidates:
            for i in range(0, len(cand)):
                i_index = member_words.index(cand[i])
                for j in range(0, len(cand)):
                    j_index = member_words.index(cand[j])
                    word_matrix[i_index][j_index] += 1
        
        return word_matrix

    #Calculate the scores for each candidate keyword
    def _score_keywords(self, member_words, word_scores, candidates, appearances):
        
        candidates_no_duplication = [] #List of each unique candidate keyword
        keyword_scores = [] #Score for each candidate keyword
        candidate_appearances = [] #Indices of where each keyword appears
        
        #For each candidate keyword, add the scores for it's member words, and if it's not already in candidatesNoDuplication, add it, it's score, and it's appearances, otherwise, just add appearance
        for i in range(0, len(candidates)):
            
            #Calculate score
            score = 0
            for word in candidates[i]:
                j = member_words.index(word)
                score += word_scores[j]

            if not candidates[i] in candidates_no_duplication:
                candidates_no_duplication.append(candidates[i])
                keyword_scores.append(score)
                candidate_appearances.append([appearances[i]])
                
            else:
                ind = candidates_no_duplication.index(candidates[i])
                candidate_appearances[ind].append(appearances[i])

        return candidates_no_duplication, keyword_scores, candidate_appearances

    #Calculate the scores of each member word and from there, each candidate keyword
    def _score_all(self, member_words, candidates, appearances):
        
        word_matrix = self._make_word_matrix(candidates, member_words)
        word_scores = [max(row) for row in word_matrix]
        candidates_no_duplication, keyword_scores, candidate_appearances = self._score_keywords(member_words, word_scores, candidates, appearances)
        
        return candidates_no_duplication, keyword_scores, candidate_appearances

    #Remove candidates that only appear once, and order candidates by score
    def _prune_candidates(self, candidates_no_duplication, keyword_scores, candidate_appearances):
        
        #Count the number of appearances for each candidate
        appear = [len(app) for app in candidate_appearances]

        #Get indices for when appearances is more than 1
        compound_inds = [index for index, value in enumerate(appear) if value > 1]
        
        #Trim lists to only contain the given indices
        candidates_no_duplication = [candidates_no_duplication[index] for index in compound_inds]
        keyword_scores = [keyword_scores[index] for index in compound_inds]
        
        #Re-order the remaining candidates
        ranked_keywords = [key for score, key in
                              sorted(zip(keyword_scores, candidates_no_duplication), key=lambda score: score[0],
                                     reverse=True)]
        ranked_scores = [score for score, key in
                              sorted(zip(keyword_scores, candidates_no_duplication), key=lambda score: score[0],
                                     reverse=True)]
        
        return ranked_keywords, ranked_scores

    #Trim a list of keywords and scores to a specified length
    def _trim_keyword_list(self, ranked_keywords, ranked_scores, n_keywords):
        if len(ranked_keywords) > n_keywords:
            ranked_keywords = ranked_keywords[0:n_keywords]
            ranked_scores = ranked_scores[0:n_keywords]
        return ranked_keywords, ranked_scores