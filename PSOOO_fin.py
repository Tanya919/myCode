import fileinput
import re
import os
import spacy
import string
import operator
from math import log2
from spellchecker import SpellChecker
from collections import Counter
import nltk
from nltk.tokenize import regexp_tokenize
from nltk.corpus import stopwords
import pandas as pd
from IPython.display import display
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import random
from rouge_score import rouge_scorer
from nltk.tokenize import word_tokenize


def sentence_num(story):
    s=[]
    for i in range(len(story)):
        s.append("S"+ str(i))
    return s

def remove_stopwords(data):
    english_stopwords = set(stopwords.words('english'))
    output_array=[]
    for sentence in data:
        temp_list=[]
        for word in sentence.split():
            if word.lower() not in english_stopwords:
                temp_list.append(word)
        output_array.append(' '.join(temp_list))
    return output_array
  
#program to count sentence length of story
def sentencelength(story):
    story_len = len(story)
    sentence_count = []
    sentence_length = []
    max_word_count = 0

    for i in range(0, story_len):
        # using regex (findall())
        # to count words in string
        res = len(re.findall(r'\w+', story[i]))
        sentence_count.append(res)

        #word count of each sentence
        #print(res)

        if(res > max_word_count):
            max_word_count = res


    for j in range(0,len(sentence_count)):
        sentenceLen = round(sentence_count[j] / max_word_count,2)
        sentence_length.append(sentenceLen)
    
    return sentence_length

#program to sentence position
def sentenceposition(story):
    sentenceLen = len(story)
    sentence_position = []

    for i in range(0, sentenceLen):
        sent_pos = round(((sentenceLen - i)/sentenceLen),2)
        sentence_position.append(sent_pos)

    return sentence_position             #sentence_position = (sentenceLen - i) / sentenceLen

#program to count numeric data in sentence of story
def numericdata(story):
    numeric_data = []
    for i in range(0, len(story)):
        # using regex (findall())
        # to count words in string
        words_count = len(re.findall(r'\w+', story[i]))
        #print(res)
        pattern = '[0-9]+'
        numeric_count = len(re.findall(pattern, story[i]))
        #print(numeric_count)
        if(words_count != 0):
            result = numeric_count/words_count
        numeric_data.append(round(result,2))
    return numeric_data

#program to find number of named entity in each sentence
def NamedEntity(story):
    NER = spacy.load("en_core_web_sm", disable=["tok2vec", "tagger", "parser", "attribute_ruler", "lemmatizer"])
    #creating a list to store the number of named entity in each sentence
    NamedEntity_=[]
    for i in range(len(story)):
        text= NER(story[i])
        #appending the number of named entity in each sentence to the list
        NamedEntity_.append(len(text.ents))
    if(max(NamedEntity_) != 0):
        NamedEntity_=[round(i/max(NamedEntity_),2) for i in NamedEntity_]
    
    return NamedEntity_

#Program to count PUNCTUATION MARKS
def specialcharecters(story):
    punctuation=[]
    for i in range(len(story)): 
        count = 0 
        for j in range(len(story[i])):  
            if story[i][j] in string.punctuation:
                count = count + 1    
        punctuation.append(count)
    
    if(max(punctuation) != 0):
        punctuation=[round(i/max(punctuation),2) for i in punctuation]
    
    return punctuation


def thematicwords(story):
    data = remove_stopwords(story)
    frequency = {}
    match_pattern = re.findall(r'\b[a-z]{3,15}\b', str(data).lower())
    for word in match_pattern:
        count = frequency.get(word,0)
        frequency[word] = count + 1
    length= len(frequency)//4
    freq_sort=sorted(frequency.items(), key=lambda x: x[1], reverse=True)
    first_data = list(map(operator.itemgetter(0), freq_sort))
    thematic_words = first_data[:length+1]
    tw=[]
    for i in range(len(data)):
        count=0
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        words = tokenizer.tokenize(data[i])
        for j in range(len(words)):
            if(words[j] in thematic_words):
                #print(words[j])
                count = count + 1
        tw.append(count)  
    if(max(tw) != 0):
        thematic_words = [round(i/max(tw),2) for i in tw]
    return thematic_words


#count no of uppercases 
def Uppercase(story):
    UpperCase = []
    for i in range(0, len(story)):
        countUpperCase = 0
        token = regexp_tokenize(story[i], "[\w']+")
        for j in token:
            if len(j) != 1 and j.isupper()==True:
                countUpperCase += 1
        UpperCase.append(countUpperCase)
    if(max(UpperCase) != 0):
            UpperCase=[round(i/max(UpperCase),2) for i in UpperCase]
    
    return UpperCase

def entropy(story):
    for i in range(len(story)):
        story[i] = story[i].lower()
    for i in range(len(story)):
        for character in string.punctuation:
             story[i] = story[i].replace(character, '')
    data = remove_stopwords(story)
    def counting(elements):
        # check if each word has '.' at its last. If so then ignore '.'
        if elements[-1] == '.':
            elements = elements[0:len(elements) - 1]

        # if there exists a key as "elements" then simply
        # increase its value.
        if elements in dictionary:
            dictionary[elements] += 1

        # if the dictionary does not have the key as "elements" 
        # then create a key "elements" and assign its value to 1.
        else:
            dictionary.update({elements: 1})
    totalCount = []
    for Sentence in data:
        dictionary = {}
        wordCount = []
        lst = Sentence.split()
        for elements in lst:
            counting(elements)
        for allKeys in dictionary:
            wordCount.append(dictionary[allKeys])
#             print ("Frequency of ", allKeys, end = " ")
#             print (":", end = " ")
#             print (dictionary[allKeys], end = " ")
#             print("-----------------") 
        totalCount.append(wordCount)
    lengthSentence = []    
    for i in range(0, len(data)):
        count = len(data[i].split())
        lengthSentence.append(count)
    def entropyCalculation(senList):
        entropy = 0
        i = 0
        length = lengthSentence[i]
        for freq in senList:
            prob = round(freq/length, 2)
            #print(-(prob * log2(prob)))
            entropy += -(prob * log2(prob))
            #print(entropy, " ")
        return entropy
    entropyTotal = []
    for i in range(0, len(totalCount)):
        #print(totalCount[i])
        ent = entropyCalculation(totalCount[i])
        entropyTotal.append(round(ent,2))    
    if(max(entropyTotal) != 0):
        entropyTotal=[round(i/max(entropyTotal),2) for i in entropyTotal]
    
    return entropyTotal

#Function to find incorrect words
#incorrect words

def incorrect(story):
    incorrectWord = []
    spell = SpellChecker()
    for i in range(len(story)):
        for character in string.punctuation:
             story[i] = story[i].replace(character, '')
    # find those words that may be misspelled
    for i in range(0, len(story)):
        l = story[i].split()
        #print(l)
        misspelled = spell.unknown(l)
        count = 0
        for word in misspelled:
            count = count + 1
        incorrectWord.append(count)
    if(max(incorrectWord) != 0):
        incorrectWord=[round(i/max(incorrectWord),2) for i in incorrectWord]
    
    return incorrectWord

#Finding and updating Parts Of Speech (POS Tags)

def postags(story):
    Postags=[]
    postags_ct = []
    for i in range(len(story)):
        ct = 0
        #tokenize the words in the text
        tokenizer = nltk.RegexpTokenizer(r"\w+")
        tokens = tokenizer.tokenize(story[i])
        #assign POS tags to each words
        pos = nltk.pos_tag(tokens)
        #Count the POS tags
        the_count = dict(Counter(tag for _, tag in pos))
        #appending the count of each pos tags in a sentence to a list
        Postags.append(the_count)
        keys = the_count.keys()
        #adding nouns and verbs together under pos category
        for i in keys:
            if(i == "NNP" or i =="NNPS" or i =="NN" or i =="NNS" or i =="VB" or i =="VBD" or i =="VBG" or i =="VBN" or i =="VBP" or i =="VBZ"):
                ct += the_count[i] 
        postags_ct.append(ct)
    if(max(postags_ct) != 0):
        postags_ct=[round(i/max(postags_ct),2) for i in postags_ct]
    return postags_ct

def tf_isf(story):
    vectorizer = TfidfVectorizer()
    vectors = vectorizer.fit_transform(story)
    #feature_names = vectorizer.get_feature_names()
    dense = vectors.todense()
    denselist = dense.tolist()
    scores=[]
    for i in range(len(denselist)):
        score=0
        for j in range(len(denselist[i])):
            score+=denselist[i][j]
        scores.append(score)
    if(max(scores) != 0):
        scores=[round(i/max(scores),2) for i in scores]
    return scores

def sentence_similarity(story):
    Tfidf_vect = TfidfVectorizer()
    vector_matrix = Tfidf_vect.fit_transform(story)
    #tokens = Tfidf_vect.get_feature_names()
    cosine_similarity_matrix = cosine_similarity(vector_matrix)
    cosines=[]
    for i in range(len(cosine_similarity_matrix)):
        cos=0
        for j in range(len(cosine_similarity_matrix[i])):
            cos= cos + cosine_similarity_matrix[i][j]
        cosines.append(cos)
    if(max(cosines) != 0):
        cosines=[round(i/max(cosines),2) for i in cosines]
    return cosines

def title_feature(story,title):
    title_features = []
    title_words = word_tokenize(title)
    length_title = len(title_words)
    for i in range(len(story)):
        score = 0
        sentence_words = word_tokenize(story[i])
        for word in sentence_words:
            if word in title_words:
                score += 1
        title_features.append(score)
    title_features=[i/length_title for i in title_features]
    return title_features

directory = 'C:/Users/LENOVO/Downloads/BBC News Summary/News Articles/business/'
files = os.listdir(directory)
#train_length = int(0.7*len(files))
#train=files[0:train_length]
train = files[100:110]
test = files[110:]
import matplotlib.pyplot as plt
#------------------------------------------------------------------------------
# TO CUSTOMIZE THIS PSO CODE TO SOLVE UNCONSTRAINED OPTIMIZATION PROBLEMS, CHANGE THE PARAMETERS IN THIS SECTION ONLY:
# THE FOLLOWING PARAMETERS MUST BE CHANGED.

def objective_function(p):
     story_text1 = "\n".join(list(fileinput.input(
         "C:/Users/LENOVO/Downloads/BBC News Summary/News Articles/business/001.txt.)))
     title = re.findall(r'^.+?\n\n\n\n', story_text1)[0]
     story_text = re.sub(r'^.+?\n\n\n\n', '', story_text1)
     #_story, highlights = process_story(story_text)
     #story = nltk.sent_tokenize(_story)
     story_token = nltk.sent_tokenize(story_text)
     story = remove_stopwords(story_token)
     ps = PorterStemmer()
     for k in range(len(story)):
         story[k] = ps.stem(story[k])
     summary = "\n ".join(list(fileinput.input(
         "C:/Users/LENOVO/Downloads/BBC News Summary/Summaries/business/"+t)))
     summary_length = summary_length = int(0.43 * len(story_token))
     df = pd.DataFrame(
         {
             'File Number ': "F" + str(t),
             'Sentence Number': sentence_num(story),
             'Sentence length': sentencelength(story),
             'Sentence Position': sentenceposition(story),
             'Numeric Data': numericdata(story),
             'Named Entity': NamedEntity(story),
             'Special Charecters': specialcharecters(story),
             'Thematic Words': thematicwords(story),
             'Upper Case': Uppercase(story),
             'Entropy': entropy(story),
             'Incorrect Word': incorrect(story),
             'POS Tags': postags(story),
             'Term Weight': tf_isf(story),
             'Cosine Similarity': sentence_similarity(story),
             'Title Feature': title_feature(story, title)
         })
     #display(df)
     feature_list = df.values.tolist()
     for i in range(len(feature_list)):
         del feature_list[i][:2]
     sentence_score = []
     for i in range(len(feature_list)):
         sum1 = 0
         for j in range(len(p)):
             sum1 += feature_list[i][j]*p[j]
         sentence_score.append(sum1)
     m = []
     sort = sorted(sentence_score, reverse=True)
     #print(sort)
     for i in range(summary_length):
         s = sentence_score.index(sort[i])
         m.append(s)
     generated_summary = ""
     for i in range(len(m)):
         generated_summary = generated_summary+story_token[m[i]]
     #print(generated_summary)
     #print(summary)
     scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
     scores = scorer.score(summary, generated_summary)
     results = {'precision': [], 'recall': [], 'fmeasure': []}
     precision, recall, fmeasure = scores['rougeL']
     # add them to the proper list in the dictionary
     results['precision'].append(precision)
     results['recall'].append(recall)
     results['fmeasure'].append(fmeasure)
    results['recall'][0]
    return results['recall'][0]

bounds=[-1,1]   # upper and lower bounds of variables
nv = 12                   # number of variables
mm = 1     
if mm == -1:
    initial_fitness = float("inf") # for minimization problem
if mm == 1:
    initial_fitness = -float("inf") # for maximization problem              # if minimization problem, mm = -1; if maximization problem, mm = 1
 
# THE FOLLOWING PARAMETERS ARE OPTIMAL.
particle_size=10        # number of particles
iterations=10        # max number of iterations
w=0.85                    # inertia constant
c1=1                    # cognative constant
c2=2                     # social constant
# END OF THE CUSTOMIZATION SECTION
#------------------------------------------------------------------------------    
class Particle:
    def __init__(self,bounds):
        self.particle_position=[]                     # particle position
        self.particle_velocity=[]                     # particle velocity
        self.local_best_particle_position=[]          # best position of the particle
        self.fitness_local_best_particle_position= initial_fitness  # initial objective function value of the best particle position
        self.fitness_particle_position=initial_fitness             # objective function value of the particle position
 
        for i in range(nv):
            self.particle_position.append(random.uniform(bounds[0],bounds[1])) # generate random initial position
            self.particle_velocity.append(random.uniform(-1,1)) # generate random initial velocity
            
    def evaluate(self,objective_function):
        self.fitness_particle_position=objective_function(self.particle_position)
        if mm == -1:
            if self.fitness_particle_position < self.fitness_local_best_particle_position:
                self.local_best_particle_position=self.particle_position                  # update the local best
                self.fitness_local_best_particle_position=self.fitness_particle_position  # update the fitness of the local best
        if mm == 1:
            if self.fitness_particle_position > self.fitness_local_best_particle_position:
                self.local_best_particle_position=self.particle_position                  # update the local best
                self.fitness_local_best_particle_position=self.fitness_particle_position  # update the fitness of the local best
 
    def update_velocity(self,global_best_particle_position):
        for i in range(nv):
            r1=random.random()
            r2=random.random()
 
            cognitive_velocity = c1*r1*(self.local_best_particle_position[i] - self.particle_position[i])
            social_velocity = c2*r2*(global_best_particle_position[i] - self.particle_position[i])
            self.particle_velocity[i] = w*self.particle_velocity[i]+ cognitive_velocity + social_velocity
 
    def update_position(self,bounds):
        
        for i in range(nv):
            
            self.particle_position[i]=self.particle_position[i]+self.particle_velocity[i]
             
            # check and repair to satisfy the upper bounds
            #if self.particle_position[i]>bounds[1]:
            #   self.particle_position[i]=bounds[1]
            # check and repair to satisfy the lower bounds
            #if self.particle_position[i] < bounds[0]:
            #     self.particle_position[i]=bounds[0]
 
 
class PSO():
    def __init__(self,objective_function,bounds,particle_size,iterations):
        
        self.w = []
        fitness_global_best_particle_position=initial_fitness
        global_best_particle_position=[]
        swarm_particle=[]
        for i in range(particle_size):
            swarm_particle.append(Particle(bounds))
        A=[]
 
        for i in range(iterations):
            for j in range(particle_size):
                swarm_particle[j].evaluate(objective_function)
 
                if mm ==-1:
                    if swarm_particle[j].fitness_particle_position < fitness_global_best_particle_position:
                        global_best_particle_position = list(swarm_particle[j].particle_position)
                        fitness_global_best_particle_position = float(swarm_particle[j].fitness_particle_position)
                if mm ==1:
                    if swarm_particle[j].fitness_particle_position > fitness_global_best_particle_position:
                        global_best_particle_position = list(swarm_particle[j].particle_position)
                        fitness_global_best_particle_position = float(swarm_particle[j].fitness_particle_position)
            for j in range(particle_size):
                swarm_particle[j].update_velocity(global_best_particle_position)
                swarm_particle[j].update_position(bounds)
 
            A.append(fitness_global_best_particle_position) # record the best fitness
        self.w = global_best_particle_position
        print('Optimal solution:', global_best_particle_position)
        print('Objective function value:', fitness_global_best_particle_position)
        print('Evolutionary process of the objective function value:')
        plt.plot(A)
        #plt.show()
        #------------------------------------------------------------------------------

#------------------------------------------------------------------------------   
# Main PSO         
optimal = PSO(objective_function,bounds,particle_size,iterations)
feature_weights=optimal.w
    
print(feature_weights)

for f in files:
    story_text1 = "\n".join(list(fileinput.input("C:/Users/LENOVO/Desktop/"+f)))
    if(len(re.findall(r'^.+?\n\n\n\n', story_text1)) == 0):
        title=""
    else:
        title = re.findall(r'^.+?\n\n\n\n', story_text1)[0]
    story_text = re.sub(r'^.+?\n\n\n\n', '', story_text1)
    #_story, highlights = process_story(story_text)
    #story = nltk.sent_tokenize(_story)
    story_token = nltk.sent_tokenize(story_text)
    story=remove_stopwords(story_token)
    ps = PorterStemmer()
    for k in range(len(story)):
      story[k]=ps.stem(story[k])
    df = pd.DataFrame(
            {
             'File Number ': "F" + "1",
             'Sentence Number': sentence_num(story),
             'Sentence length' : sentencelength(story), 
             'Sentence Position': sentenceposition(story), 
             'Numeric Data' : numericdata(story),
             'Named Entity' : NamedEntity(story),
             'Special Charecters' : specialcharecters(story),
             'Thematic Words': thematicwords(story),
             'Upper Case' : Uppercase(story),
             'Entropy' : entropy(story),
             'Incorrect Word' : incorrect(story),
             'POS Tags': postags(story),
             'Term Weight' : tf_isf(story),
             'Cosine Similarity' : sentence_similarity(story),
             'Title Feature':title_feature(story,title)
             })  
    #display(df)
    #df.to_csv('Features Extracted from stories.csv',index=False)
    feature_list= df.values.tolist()
    for i in range(len(feature_list)):
        del feature_list[i][:2]
    summary_length =int( 0.43 * len(story_token))
    sentence_score = []
    for i in range(len(feature_list)):
        sum1 = 0
        for j in range(len(feature_weights)):
            sum1 += feature_list[i][j]*feature_weights[j]
        sentence_score.append(sum1)
    m=[]
    sort = sorted(sentence_score, reverse=True)
    #print(sort)
    for i in range(summary_length):
        s=sentence_score.index(sort[i])
        m.append(s)
    generated_summary=""
    for i in range(len(m)):
        generated_summary = generated_summary+story_token[m[i]]
    results1 = {'precision': [], 'recall': [], 'fmeasure': []}
    # add them to the proper list in the dictionary
    results1['precision'].append(precision)
    results1['recall'].append(recall)
    results1['fmeasure'].append(fmeasure)

    p1.append(precision)
    r1.append(recall)
    f1.append(fmeasure)

    results2 = {'precision': [], 'recall': [], 'fmeasure': []}
    # add them to the proper list in the dictionary
    results2['precision'].append(precision2)
    results2['recall'].append(recall2)
    results2['fmeasure'].append(fmeasure2)

    p2.append(precision2)
    r2.append(recall2)
    f2.append(fmeasure2)

    resultsL = {'precision': [], 'recall': [], 'fmeasure': []}
    # add them to the proper list in the dictionary
    resultsL['precision'].append(precisionL)
    resultsL['recall'].append(recallL)
    resultsL['fmeasure'].append(fmeasureL)

    pl.append(precisionL)
    rl.append(recallL)
    fl.append(fmeasureL)
    print("\n\n") 
# print("Mean ", np.mean(r1))
final_results.loc[len(final_results.index)] = [np.mean(p1), np.mean(r1), np.mean(f1) , np.mean(p2), np.mean(r2),  np.mean(f2) ,  np.mean(pl),  np.mean(rl),  np.mean(fl)]
final_results.to_csv('C:/Users/LENOVO/Downloads/Bus_Results_' + str(b) + '.csv')
#         print(scores)
    
    

def get_summary(doc):
    story_text1 = "\n".join(list(fileinput.input("C:/Users/LENOVO/Desktop/"+doc)))
    if(len(re.findall(r'^.+?\n\n\n\n', story_text1)) == 0):
        title=""
    else:
        title = re.findall(r'^.+?\n\n\n\n', story_text1)[0]
    story_text = re.sub(r'^.+?\n\n\n\n', '', story_text1)
    #_story, highlights = process_story(story_text)
    #story = nltk.sent_tokenize(_story)
    story_token = nltk.sent_tokenize(story_text)
    story=remove_stopwords(story_token)
    ps = PorterStemmer()
    for k in range(len(story)):
      story[k]=ps.stem(story[k])
    df = pd.DataFrame(
            {
             'File Number ': "F" + "1",
             'Sentence Number': sentence_num(story),
             'Sentence length' : sentencelength(story), 
             'Sentence Position': sentenceposition(story), 
             'Numeric Data' : numericdata(story),
             'Named Entity' : NamedEntity(story),
             'Special Charecters' : specialcharecters(story),
             'Thematic Words': thematicwords(story),
             'Upper Case' : Uppercase(story),
             'Entropy' : entropy(story),
             'Incorrect Word' : incorrect(story),
             'POS Tags': postags(story),
             'Term Weight' : tf_isf(story),
             'Cosine Similarity' : sentence_similarity(story),
             'Title Feature':title_feature(story,title)
             })  
    #display(df)
    #df.to_csv('Features Extracted from stories.csv',index=False)
    feature_list= df.values.tolist()
    for i in range(len(feature_list)):
        del feature_list[i][:2]
    summary_length =int( 0.43 * len(story_token))
    sentence_score = []
    for i in range(len(feature_list)):
        sum1 = 0
        for j in range(len(feature_weights)):
            sum1 += feature_list[i][j]*feature_weights[j]
        sentence_score.append(sum1)
    m=[]
    sort = sorted(sentence_score, reverse=True)
    #print(sort)
    for i in range(summary_length):
        s=sentence_score.index(sort[i])
        m.append(s)
    generated_summary=""
    for i in range(len(m)):
        generated_summary = generated_summary+story_token[m[i]]
    print("-------------------------------The story given is:----------------------------------",)
    print()
    print(story_text1)
    print("----------------------------The summary generated is:----------------------------------")
    print()
    print(generated_summary)



        
        
        
        
        
                
        
    



