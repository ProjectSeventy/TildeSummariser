from spacy.tokens.span import Span
from nltk.corpus import wordnet

from tilde.utils.components.base_components import RedundancyDetector

    
class SimpleRTE(RedundancyDetector):
    
    """
    A class implementing SimpleRTE
    
    This is a simple redundancy detection algorithm
    """
    
    def __init__(
        self,
        entity_weight: int = 2,
        element_weight: int = 1,
        threshold: float = 0.8,
    ):
        """
        Initialises the SimpleRTE class

        Args:
            entity_weight: An int amount to weight the entity score
            element_weight: An int amount to weight the element score
            threshold: A float threshold of where to indicate redundancy
        """
        
        self.entity_weight = entity_weight
        self.element_weight = element_weight
        self.threshold = threshold
        
        self.subject_object_list = [
            "csubj",
            "csubjpass",
            "nsubj",
            "nsubjpass",
            "dobj",
            "pobj",
            "oprd",
            "obj",
            "neg"
        ]
        self.verb_list = [
            "ROOT",
            "acomp",
            "pcomp",
            "xcomp"
        ]

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
        
        #Split both texts into elements and entities
        text_elems, hypo_elems = self._get_all_elements(text, hypo)
        text_ents, hypo_ents = self._get_all_named_entities(text, hypo)
        
        #Calculate the negation, entity overlap, and element overlap scores for the texts
        neg_score = self._calculate_overall_negation_score(text, hypo)
        elem_score = self._calculate_element_score(text_elems, hypo_elems)
        ent_score = self._calculate_entity_score(text_ents, hypo_ents)
        
        #Overall score is the weighted average of the entity and element scores, minus the negation score.
        overall_score = (((elem_score*self.element_weight) + (ent_score*self.entity_weight) ) / (self.element_weight + self.entity_weight)) - neg_score

        #Return true if score is greater than the threshold
        return overall_score > self.threshold
    
    def _get_children(self, token, child_list, strin):
        """Return a list of words that are children to the given one in the dependency tree, that are subjects or objects"""

        for child in token.children:
            if child.dep_ in self.subject_object_list:
                strin.append(child.lemma_.lower())
                child_list.append(set(strin))
                strin = []
            else:
                nstrin = []
                child_list = self._get_children(child, child_list, nstrin)

        return child_list

    def _get_elements(self, text):
        """Get a list of verb, subject/object pairs from a given text"""

        text_elems = []
        for token in text:
            #If the token is a verb, get a list of it's subjects and objects
            if token.dep_ in self.verb_list:
                for lis in self._get_children(token, [], []):
                    lis.add(token.lemma_.lower())

                    if len(lis) == 2:
                        text_elems.append(lis)

        return text_elems

    def _get_all_elements(self, text, hypo):
        """Get all the elements from two strings - hypothesis and text"""

        text_elems = self._get_elements(text)
        hypo_elems = self._get_elements(hypo)
        
        return text_elems, hypo_elems

    def _get_named_entities(self, text):
        """Get a list of the named entities in a text"""

        text_ents = []

        if text.ents:
            for ent in text.ents:
                text_ents.append(ent.text.lower())
        
        return text_ents

    def _get_all_named_entities(self, text, hypo):
        """Get all the named entities from two strings - hypothesis and text"""

        text_ents = self._get_named_entities(text)
        hypo_ents = self._get_named_entities(hypo)

        return text_ents, hypo_ents

    def _calculate_negation_score(self, text):
        """Count the number of negatory words in a text"""

        neg_score = 0
        
        for token in text:
            if token.dep_ == "neg":
                neg_score += 1
        
        return neg_score

    def _calculate_overall_negation_score(self, text, hypo):
        """Calculate the negation score for two given texts"""
        
        text_neg_score = self._calculate_negation_score(text)
        hypo_neg_score = self._calculate_negation_score(hypo)
        #Negation score is the difference in negation score of both texts
        neg_score = abs(text_neg_score - hypo_neg_score)
        return neg_score

    def _get_lemma_names_from_synset(self, syns):
        """Return a list of lemmas from a given synset"""

        synset = [ele.lemmas() for ele in syns]
        lems = []
        for lem_list in synset:
            for lemma in lem_list:
                lems.append(lemma.name().split(".")[0])
        return lems

    def _get_all_related_lemma_names(self, word):
        """Gets a list of all lemmas related to a given word"""
        
        syns = wordnet.synsets(word)
        synset = self._get_lemma_names_from_synset(syns)
        
        hyponyms = []
        hypernyms = []
        
        for syn in syns:
            hyponyms = syn.hyponyms()
            hyponyms = self._get_lemma_names_from_synset(hyponyms)
            hypernyms = syn.hypernyms()
            hypernyms = self._get_lemma_names_from_synset(hypernyms)
        
        synset += hyponyms + hypernyms
        
        return synset
    
    def _calculate_element_score(self, text_elems, hypo_elems):
        """Calculate the overlap score of two lists of elements"""
        
        elem_crossover = 0
        
        for elem in hypo_elems:
            if elem in text_elems:
                elem_crossover += 1
            else:
                #If an element is not present, gather all related lemmas for each word in the element
                it = iter(elem)
                synset = self._get_all_related_lemma_names(next(it))
                synset2 = self._get_all_related_lemma_names(next(it))
                
                for word1 in synset:
                    for word2 in synset2:
                        if {word1, word2} in text_elems:
                            elem_crossover += 1
                            break;

        elem_score = (elem_crossover + 1) / (len(hypo_elems)+1)
        return elem_score

    def _calculate_entity_score(self, text_ents, hypo_ents):
        """Calculate the overlap score of two lists of named entities"""

        ent_score = len(hypo_ents)

        for ent in hypo_ents:
            if not ent in text_ents:
                ent_score -= 1

        ent_score = (ent_score + 1) / (len(hypo_ents) + 1)

        return ent_score