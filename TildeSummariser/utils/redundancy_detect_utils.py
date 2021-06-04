from nltk.corpus import wordnet

subject_object_list = ["csubj", "csubjpass", "nsubj", "nsubjpass", "dobj", "pobj", "oprd", "obj", "neg"]
verb_list = ["ROOT", "acomp", "pcomp", "xcomp"]
entity_weight = 2

#Return a list of words that are children to the given one in the dependency tree, that are subjects or objects
def get_children(token, child_list, strin):
    
    for child in token.children:
        #If a given child is subj/obj, add it to the list and return
        if child.dep_ in subject_object_list:
            strin.append(child.lemma_.lower())
            child_list.append(set(strin))
            strin = []
        #Otherwise check the element's children
        else:
            nstrin = []
            child_list = get_children(child, child_list, nstrin)
    return child_list

#Get a list of verb, subject/object pairs from a given text
def _get_elements(text):
    text_elems = []
    for token in text:
        #If the token is a verb, get a list of it's subjects and objects
        if token.dep_ in verb_list:
            for lis in get_children(token, [], []):
                lis.add(token.lemma_.lower())
                #If pair is correctly formed, add it to the list of elements. Without this line, I was getting several single elements, and couldn't figure out why.
                if len(lis) == 2:
                    text_elems.append(lis)
    return text_elems

#Get all the elements from two strings - hypothesis and text
def get_elements(text, hypo):
    
    text_elems = _get_elements(text)
    hypo_elems = _get_elements(hypo)
    
    return text_elems, hypo_elems

#Get a list of the named entities in a text
def _get_named_entities(text):
    text_ents = []

    if text.ents:
        for ent in text.ents:
            text_ents.append(ent.text.lower())
    
    return text_ents

#Get all the named entities from two strings - hypothesis and text
def get_named_entities(text, hypo):
    
    text_ents = _get_named_entities(text)
    hypo_ents = _get_named_entities(hypo)

    return text_ents, hypo_ents

#Count the number of negatory words in a text
def _calculate_negation_score(text):
    neg_score = 0
    
    for token in text:
        if token.dep_ == "neg":
            neg_score += 1
    
    return neg_score

#Calculate the negation score for two given texts
def calculate_negation_score(text, hypo):
    
    text_neg_score = _calculate_negation_score(text)
    hypo_neg_score = _calculate_negation_score(hypo)
    neg_score = abs(text_neg_score - hypo_neg_score) #Negation score is the difference in negation score of both texts
    return neg_score

#Return a list of lemmas from a given synset
def _get_lemma_names_from_synset(syns):
    synset = [ele.lemmas() for ele in syns]
    lems = []
    for lem_list in synset:
        for lemma in lem_list:
            lems.append(lemma.name().split(".")[0])
    return lems

#Gets a list of all lemmas related to a given word
def get_all_related_lemma_names(word):
    
    syns = wordnet.synsets(word)
    #Get all lemmas of synonyms
    synset = _get_lemma_names_from_synset(syns)
    
    hyponyms = []
    hypernyms = []
    
    for syn in syns:
        #For each meaning of the word, get all lemmas of hyponyms and hypernyms
        hyponyms = syn.hyponyms()
        hyponyms = _get_lemma_names_from_synset(hyponyms)
        hypernyms = syn.hypernyms()
        hypernyms = _get_lemma_names_from_synset(hypernyms)
    
    synset += hyponyms + hypernyms
    
    return synset

#Calculate the overlap score of two lists of elements
def calculate_element_score(text_elems, hypo_elems):
    
    elem_crossover = 0
    
    #For each element, first increase score for each hypothesis element also present in the text
    for elem in hypo_elems:
        if elem in text_elems:
            elem_crossover += 1
        else:
            #If an element is not present, gather all related lemmas for each word in the element
            it = iter(elem)
            synset = get_all_related_lemma_names(next(it))
            synset2 = get_all_related_lemma_names(next(it))
            
            #Check each combination of lemmas against the text elements, and if any are present, increase the element score and break out
            for word1 in synset:
                for word2 in synset2:
                    if {word1, word2} in text_elems:
                        elem_crossover += 1
                        break;

    #Element score calculated as proportion of hypothesis elements also present in the text. 1 added to numerator and denominator to prevent division by zero
    elem_score = (elem_crossover + 1) / (len(hypo_elems)+1)
    return elem_score

#Calculate the overlap score of two lists of named entities
def calculate_entity_score(text_ents, hypo_ents):
    
    #First set score to the total number of entities in the hypothesis
    ent_score = len(hypo_ents)
    #Subtract 1 from the score for each entity not also in the list of text entities
    for ent in hypo_ents:
        if not ent in text_ents:
            ent_score -= 1
    #Entity score caluclated as proportion of hypothesis entities also present in the text. 1 added to numerator and denominator to prevent division by zero
    ent_score = (ent_score + 1) / (len(hypo_ents) + 1)

    return ent_score

#Check if a given text entails another
def check_entailment(text, hypo):
    
    #Split both texts into elements and entities
    text_elems, hypo_elems = get_elements(text, hypo)
    text_ents, hypo_ents = get_named_entities(text, hypo)
    
    #Calculate the negation, entity overlap, and element overlap scores for the texts
    neg_score = calculate_negation_score(text, hypo)
    elem_score = calculate_element_score(text_elems, hypo_elems)
    ent_score = calculate_entity_score(text_ents, hypo_ents)
    
    #Overall score is the weighted average of the entity and element score, - the negation score. The entity score is weighted further (by a factor of 2) as experimentally shown to be more important
    overall_score = ((elem_score + (ent_score*entity_weight) ) / (1 + entity_weight)) - neg_score

    #Return true if score is greater than 0.8
    return overall_score > 0.8