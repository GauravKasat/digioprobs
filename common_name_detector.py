from fuzzywuzzy import fuzz #type: ignore
#from phonetics import metaphone
from sklearn.feature_extraction.text import TfidfVectorizer #type: ignore
from sklearn.cluster import KMeans #type: ignore
from langdetect import detect #type: ignore 
from collections import Counter #type: ignore
import numpy as np #type: ignore
import faiss #type: ignore
import jellyfish #type: ignore


import pandas as pd
df = pd.read_excel('Names.xlsx')
name_dataset = df['cname'].to_list()
fnames = df['fname'].to_list()
lnames = df['lname'].to_list()
frequency_dict = Counter(name_dataset)
fname_frequency_dict = Counter(fnames)
lname_frequency_dict = Counter(lnames)



def calculate_entropy(name):
    probs = [name.count(c) / len(name) for c in set(name)]
    entropy = -sum(p * np.log2(p) for p in probs)
    return entropy / np.log2(len(name))

# Function to calculate rarity score
def calculate_rare_name_score(name, dataset,frequency_dict):
    if len(name.split())==2:
        f_name,l_name = name.split()[0],name.split()[1]
    else:
        f_name = name
        l_name = name
    print(f_name,l_name)
    frequency_score =  1 / (1 + frequency_dict.get(f_name, 0))
    frequency_score1 = 1 / (1 + fname_frequency_dict.get(f_name, 0))
    frequency_score2 = 1 / (1 + lname_frequency_dict.get(l_name, 0))

    print(frequency_score,frequency_score1,frequency_score2)
    if frequency_score<=0.6:
        return False

    if frequency_score1!=1 and frequency_score1>=0.5:
        if frequency_score2!=1 and frequency_score2>=0.6:
            return True


    fuzzy_scores = [fuzz.ratio(name, dataset_name) for dataset_name in dataset]
    fuzzy_match_score = (max(fuzzy_scores) / 100)
    print("fuzzy_match_score " + str(fuzzy_match_score)  )

    if fuzzy_match_score>=0.5:
        return False

    
    sound_Scores = [fuzz.ratio(jellyfish.soundex(name), jellyfish.soundex(dataset_name)) for dataset_name in dataset]

    sound_match_score = (max(sound_Scores) / 100)
    print("sound_match_score " + str(sound_match_score) )


    if sound_match_score>0.3:
        return False


    entropy_score = calculate_entropy(name)
    print("entropy_score " + str(entropy_score) )




    vectorizer = TfidfVectorizer()
    c = name_dataset + fnames + lnames
    embeddings = vectorizer.fit_transform(c).toarray()
    name_embedding = vectorizer.transform([name]).toarray()

    
    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(embeddings)
    distances, _ = index.search(name_embedding, k=1) 
    clustering_score = distances[0][0]
    print("clustering_score: " + str(clustering_score))
    if clustering_score>=0.49:
        return True  

    
    ## some lang detection which can detect hinglish content robustly.
    try:
        language = detect(name)
        foreign_score = 1 if language != 'en' and language != 'hi' else 0  # Assume 'en' and 'hi' as common
    except:
        foreign_score = 0
    #if foreign_score==1:
    #    return True

    #print("Foreign Score: " + str(foreign_score)  )

    # aggregate score
    return False

# Example Usage
input_name = input("Type the name of the person: ").lower()
rarity_score = calculate_rare_name_score(input_name, name_dataset,frequency_dict)
if rarity_score:
    print("Rare\n")
else:
    print("not Rare\n")
#print(f"Rarity Score for '{input_name}': {rarity_score:.2f}")