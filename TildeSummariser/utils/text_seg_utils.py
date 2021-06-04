import spacy
import math

#Resize the docs in the input to each be one unit
def resize_units(content):
    n=1
    length = len(content)
    if length > 300:
        n = 3
        if length > 500:
            n = 5
        if length > 1000:
            n =10
        if length > 3000:
            n = 30
        if length > 5000:
            n = 50
        new_content = []
        for i in range(0, length, n):
            if i+n < len(content):
                new_content.append(spacy.tokens.Doc.from_docs(content[i:i+n]))
            else:
                new_content.append(spacy.tokens.Doc.from_docs(content[i:]))
        
        content = new_content
    return content

#Get frequency of lemmas in a specific doc
def get_lemma_freqs(content):
    all_lemma_freqs = []
    for unit in content:
        lemma_freq = {}
        for token in unit:
            if not (token.is_stop or token.is_punct or token.is_space):
                lemma = token.lemma_
                lemma_freq[lemma] = lemma_freq.get(lemma, 0) + 1
        all_lemma_freqs.append(lemma_freq)
    
    return all_lemma_freqs

#Merge several lemma frequency dictionaries into one
def merge_lemma_dictionaries(all_lemma_freqs):
    lemma_freqs = {}
    for i in range(0, len(all_lemma_freqs)):
        for word in all_lemma_freqs[i].keys():
            lemma_freqs[word] = lemma_freqs.get(word, [0]*i)
            lemma_freqs[word].append(all_lemma_freqs[i][word])
        for word in lemma_freqs.keys():
            if word not in all_lemma_freqs[i].keys():
                lemma_freqs[word].append(0)
    
    return lemma_freqs

#Generate a matrix of similarity scores between all pairs of units
def generate_unit_similarity_matrix(num_units, lemma_freqs):
    #Construct an empty matrix representing each unit occuring with itself
    para_matrix = []
    for i in range(0, num_units):
        para_row = [0] * num_units
        para_matrix.append(para_row)

    #For each pair of units, fill their cells with the cosine difference between their lemma freqs
    for i in range(0, num_units):
        for j in range(i, num_units):
            #Calculate cosine similarity
            ab = []
            asq = []
            bsq = []
            for lemma in lemma_freqs.keys():
                freqs = lemma_freqs.get(lemma)
                ab.append(freqs[i] * freqs[j])
                asq.append(freqs[i] * freqs[i])
                bsq.append(freqs[j] * freqs[j])
            num = sum(ab)
            denom = (math.sqrt(sum(asq))) * (math.sqrt(sum(bsq)))
            if denom == 0:
                sim = 0
            else:
                sim = num/denom
            
            #Populate mirrored and normal half of matrix
            para_matrix[i][j] = sim
            para_matrix[j][i] = sim
    
    return para_matrix

#Calculate the density of the unit matrix with the specified divisions
def calculate_density_from_divisions(para_matrix, divisions):
    divisions = sorted(divisions)
    nums = []
    denoms = []
    prev_div = 0
    #For each cell in each division, add the value to the numerator and 1 to the denominator
    for div in divisions:
        num = 0
        denom = 0
        for i in range(prev_div, div):
            for j in range(prev_div, div):
                num += para_matrix[i][j]
                denom += 1
        nums.append(num)
        denoms.append(denom)
    return (sum(nums)/sum(denoms))

#Get an ordered list of the most optimal division of the unit similarity matrix
def get_optimal_division_list(para_matrix, num_units):
    divisions = [num_units]
    density = calculate_density_from_divisions(para_matrix, divisions.copy())
    density_history = [density]
    while(True):
        densities = []
        new_divs = []
        if len(divisions) == num_units:
            break
        if len(divisions) == 150:
            break
        for i in range(1, num_units):
            if i not in divisions:
                densities.append(calculate_density_from_divisions(para_matrix, divisions.copy() + [i]))
                new_divs.append(i)
        if max(densities) > density:
            density = max(densities)
            density_history.append(density)
            division = new_divs[densities.index(density)]
            divisions.append(division)
        else:
            break
    
    return divisions, density_history

#Get the divisions corresponding with the optimal segmentation
def get_optimal_divisions(divisions, density_history):
    delta_density = density_history[1:]
    for i in range(0, len(delta_density)):
        delta_density[i] = delta_density[i] - density_history[i]
    
    mean_delta_density = sum(delta_density)/len(delta_density)
    variance_delta_density = sum([(val-mean_delta_density)*(val-mean_delta_density) for val in delta_density])/(len(delta_density)+1)
    threshold = mean_delta_density + (1.2*variance_delta_density)
    
    for i in range(len(delta_density)-1, -1, -1):
        if delta_density[i] > threshold:
            divisions = divisions[0:i+2]
            break
    
    return sorted(divisions)

#Get the text of each segment, based on a given set of divisions
def get_segments_from_divisions(content, divisions, unit_lengths):
    segments = []
    segments_sented = []
    sents = list(content.sents)
    last = 0
    sent_last = 0
    for i in range(0, len(divisions)):
        num_sents = sum(unit_lengths[last:divisions[i]])
        sent_list = sents[sent_last:sent_last+num_sents]
        sent_last += num_sents
        start = sent_list[0].start
        end = sent_list[-1].end
        segments.append(content[start:end])
        segments_sented.append(sent_list)
        last = divisions[i]
          
    return segments, segments_sented

#Get the optimal segments from a given text
def get_segments(content):
    content = resize_units(content)
    
    #Get the number of sentences in each unit, and the number of units
    unit_lengths = [len(list(doc.sents)) for doc in content]
    num_units = len(unit_lengths)
    
    #Generate a lemma frequency dictionary for each unit
    all_lemma_freqs = get_lemma_freqs(content)
    
    #Merge all the dicitonaries into one with a list of frequencies, one for each unit
    lemma_freqs = merge_lemma_dictionaries(all_lemma_freqs)
    
    #Merge all the units into one doc
    content = spacy.tokens.Doc.from_docs(content)
    
    #Generate unit similarity matrix
    para_matrix = generate_unit_similarity_matrix(num_units, lemma_freqs)
    
    #Get ordered list of optimal divisions, and their densities
    divisions, density_history = get_optimal_division_list(para_matrix, num_units)
    
    #Find optimal divisions for segmentation
    divisions = get_optimal_divisions(divisions, density_history)
    
    #Return optimal segments as span and list of sentences
    return get_segments_from_divisions(content, divisions, unit_lengths)