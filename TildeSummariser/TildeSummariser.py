from TildeSummariser.utils import get_segments, get_summary_sentences

import spacy

class TildeSummariser():
    
    def __init__(
        self,
        curve_coefficients = (
            0.00025,
            -0.0325,
            0.705,
            34.785,
        )
    ):
        self.curve_coefficients = curve_coefficients
        self.nlp = spacy.load("en_core_web_sm")
    
    def _get_segment_length_from_curve(self, midpoint_percentage):
        num_sentences = 0
        for i in range(len(self.curve_coefficients), 0, step=-1):
            num_sentences += pow(midpoint_percentage, i)*self.curve_coefficients[i]
        num_sentences = round(num_sentences)
        return num_sentences

    def summarise(self, content, n_segment, n_total, dynamic_flag = False):
        #pre-process
        content = list(self.nlp.pipe(content))
        content = [doc for doc in content if not (len(doc) == 1 and doc[0].text == '\n')]
        
        #Segment
        segments, segments_sented = get_segments(content)
        
        all_summary_sentences = []
        len_prev_seg = 0
        len_all_seg = sum([len(seg) for seg in segments_sented])
        for i in range(0, len(segments)):
            #If synamic segment lengths, calculate length
            if dynamic_flag:
                #Find the last sentence
                last_sent = len(segments_sented[i]) + len_prev_seg
                len_prev_seg = last_sent
                #Get the midpoint of the segment as a percentage of the full text
                perc = ((last_sent - (len(segments_sented[i])/2))/len_all_seg)*100
                #Calculate the length according to the curve
                n_segment = self._get_segment_length_from_curve(perc)
            #Get the summary sentences for the segment
            all_summary_sentences += get_summary_sentences(segments[i], segments_sented[i], n_segment)
        #Convert all summaries to one doc and generate a summary of them
        summary_sentences_as_doc = list(nlp.pipe(all_summary_sentences))
        summary_sentences_as_doc = spacy.tokens.Doc.from_docs(summary_sentences_as_doc)
        final_summary_sentences = get_summary_sentences(summary_sentences_as_doc, list(summary_sentences_as_doc.sents), n_total)
        return final_summary_sentences