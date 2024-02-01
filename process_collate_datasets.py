import json
import nltk
from textblob import TextBlob
import torch
from sklearn.metrics.pairwise import cosine_similarity
import string
from nltk.corpus import stopwords
STOP_WORDS = set(stopwords.words('english'))
STOP_WORDS.update(['.', '?', '!', ',', '(', ')' , '[',']'])
from spellchecker import SpellChecker
import re
import threading
import matplotlib.pyplot as plt
import pyphen
from sklearn.feature_extraction.text import TfidfVectorizer
import csv
import collections
import numpy as np
import ast
from transformers import BertTokenizer, BertModel, BartTokenizer, BartForConditionalGeneration
NUMTHREADS = 20


#Load pre-trained BERT model and tokenizer
tokenizer_bert = BertTokenizer.from_pretrained('bert-base-uncased')
model_bert = BertModel.from_pretrained('bert-base-uncased')

#Load pre-trained BART model and tokenizer
tokenizer_bart = BartTokenizer.from_pretrained('facebook/bart-large-cnn')
model_bart = BartForConditionalGeneration.from_pretrained('facebook/bart-large-cnn')


#Global grammar score dictionary to be used by threads to speed up grammar checking
grammarDict = {}

def main():

    
    keepGoing = True
    while keepGoing:
        choice = int(input("Enter 1 to process training data, 2 to process test data: "))
        if choice ==1 or choice ==2:
            keepGoing =False
        
    if choice == 1:
        filename= 'C:\\Users\\josep\\OneDrive\\Desktop\\ThesisWork\\CRS_Recommendation_Classifiaction\\eredial.json'
        Inspired_path = "C:\\Users\\josep\\OneDrive\\Desktop\\ThesisWork\\CRS_Recommendation_Classifiaction\\inspired_train.tsv"
        
    else:
        filename= 'C:\\Users\\josep\\OneDrive\\Desktop\\ThesisWork\\CRS_Recommendation_Classifiaction\\eredial_test.json'
        Inspired_path = "C:\\Users\\josep\\OneDrive\\Desktop\\ThesisWork\\CRS_Recommendation_Classifiaction\\inspired_test.tsv"
        
    
    spell_check =  SpellChecker()


    idList,wholeConv,preservedConversation = collectEredialData(filename)
    idList,wholeConv, movies,preservedConversation = collectInspiredData(Inspired_path, idList, wholeConv,preservedConversation)
    
    goodConvs = combineConsecutiveSpeakerSentences(idList,preservedConversation)
    
    #Quality Factor calculations
    lenScores = scoreLength(idList,wholeConv)
    readScores = scoreReadability(idList,wholeConv)
    wiScores = scoreWordImportance(idList,wholeConv)
    repScores = scoreRepitition(idList, wholeConv)
    subScores = scoreSubjectivity(idList, wholeConv)
    polScores = scorePolarity(idList, wholeConv)
    gramScores = threadProcessing(idList, wholeConv,movies)
    faScores = scoreFeatureAppearance(idList, wholeConv)
    dataDict = crunchStats(lenScores,readScores,wiScores,repScores,subScores,polScores,gramScores, faScores)
    
    '''
    #Write stats of quality factor to a file
    writeQFStats(dataDict, choice)

    #Write dataset to file
    writeCombinedData(idList,wholeConv, lenScores, readScores, wiScores,repScores,subScores,polScores,gramScores,faScores, goodConvs, choice)
    '''

    return
 


###############################################################################
#Scoring functions to add context labels to the dataset                       #
#Factors of Explainability:                                                   #
#                                                                             #
#1. Length: defined as the number of words after removing stop words          #
#since the length of explanations may influence how users perceive            #
#the explanations.                                                            #
#                                                                             #
#2. Readability: How readable the text is. calculated based on the            #
#Flesch-Kincaid readability test higher scores indicate that the material is  #
#easier to read in the Flesch reading ease test.                              #
#                                                                             #
#3. Word importance calculated by inverted term frquency and then adding      #
#the individual word impotance scores                                         #
#                                                                             #
#4.Repetition refers to how many duplicate words an explanation has.          #
#Calculated by counting the number of repeatd words once stop words have      #
#been removed                                                                 #
#                                                                             #
#5.Subjectivity reflects if the explanation contains personal opinion,        #
#emotion, and/or judgment. Calculated with textblob                           #                                                                             #
#6. Polarity is the confidence level that explanations are positive or        #
#negative. Calculated with textblob                                           #
#                                                                             #
#7. Grammatical Correctness measures the grammar quality of the explanation.  # 
#Calculated by counting the number of spelling / grammar errors in an         #
#explanation                                                                  #
#                                                                             #
#8. Feature appearance measures if an explanation captures item features.     #
#Calculated by [insert here]                                                  #  
###############################################################################



##########################################################################################
# LENGTH Functions
#Takes in a string sentence, removes stop words, calculated the length of the sentence. 
def findSentenceLength(sentence):
    cleanedSentence = nltk.word_tokenize(sentence)
    length = len(cleanedSentence)
    return length

#Finds max, min, average lengths of recommendations in the dataset
def findLengthStats(idList, convList, ):

    lengthList = []
    stdDev = 0
    #Iterate over conversation, find lengths of the recommendations
    for id in idList:
        curSentence = " ".join(convList[hash(id)][0])
        lengthList.append(findSentenceLength(curSentence))


    high = max(lengthList)
    low = min(lengthList)
    average = sum(lengthList) / len(lengthList)
    sortedLengths = sorted(lengthList)
    middle = len(sortedLengths)//2
    if len(lengthList) %2 ==0:
        median = (sortedLengths[middle-1] + sortedLengths[middle]) /2
    else:
        median = sortedLengths[middle]
    #Get standard deviation
    for lengthVal in lengthList:
        # (x_i - mean) ^2
        stdDev += (lengthVal-average)**2 

    # sqrt( (sumation x_i - mean^2)/size )
    stdDev=  (stdDev/len(lengthList))**.5
    
    '''
    plt.hist(sortedLengths, bins = 10, color='blue', edgecolor='black')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.title('Histogram of Data')

    # Display the plot
    plt.show()
    '''

    return high, low, average, stdDev

#Scores a recommendation where lengths that deviate away from the average are penalized. 
def scoreLength(idList,wholeConv):
    scores = []

    #Length is scaled by the lengths of conversations in the corpus. Get stats before crunching length scores
    high,low,average,stdDev = findLengthStats(idList, wholeConv)
    
    #Loop over conversations, score length and add to list.
    for count, id in enumerate(idList):
        curSentence = " ".join(wholeConv[hash(id)][0])
        scores.append(calcSentenceLengthScore(curSentence, average, stdDev))
    return scores
    
def calcSentenceLengthScore(sentence,average,stdDev):

    #Penalties for sentence length
    maxPenalty = .5
    leftPenalty = 1.5           #Penalize shorter sentence more than verbose ones
    rightPenalty = 1.35         #Penalize longer sentences, but less than short ones
    maxStdDev=2.5   

    #Get length of the sentence:
    length = findSentenceLength(sentence)

    #Get Z score for the data
    zScore = (length-average)/stdDev

    #If sentences are more than 2.5 stdDevs from the mean
    #Assign 0 for length score. Either too long or too short
    if abs(zScore) > maxStdDev:
        return 0

    #Get penalty for how far a sentence is from the mean
    penalty = min(maxPenalty, abs(zScore))

    #Apply penalty for shorter sentences
    if zScore < 0:
        penalty *= leftPenalty
    elif zScore > 0:
        penalty *= rightPenalty

    #Get score value
    score = 1 -penalty

    #return score in range [0,1]
    return max(0, min(1, score))

#END LENGTH Functions  
##########################################################################################


##########################################################################################
#READABILITY Functions
#Calculates Flesch Kincaid Reading Ease Score
#Takes in a recommendation in the form of a list with each element being a 
#Sentence in the recommendation
def scoreReadability(idList,wholeConv):
    scores = []
    #loop through conversation, score readability, record score in list
    for count, id in enumerate(idList):
        curList = wholeConv[hash(id)][0]
        scores.append(scoreSentenceReadability(curList))
    return scores

def scoreSentenceReadability(sentenceList):
    #Formula for score is Flesch Reading Ease Score = 206.835 − 1.015 × ( Total Words / Total Sentences ) − 84.6 × ( Total Syllables / Total Words )
    totalSentences = len(sentenceList)
    
    #Get total words and syllables
    totalWords = 0
    totalSyllables = 0
    #Iterate over each sentence in the conversation
    for sentence in sentenceList:
        words = sentence.split() #Split sentence into words
        for word in words:
            totalWords+=1
            totalSyllables += countSyllables(word)
    #Calculate readability score
    score = 206.835 -(1.015 * (totalWords/totalSentences)) -(84.6 * (totalSyllables/totalWords))
    return score
    
    
#Count the number of syllables in word
def countSyllables(word):
    dictionary = pyphen.Pyphen(lang='en_US')
    return max(1, len(dictionary.positions(word))+1)

#END READABILITY Functions
##########################################################################################

##########################################################################################
#Word importance functions

def scoreWordImportance(idList, wholeConv):
    scoresList = []             #Will hold scores for each conversation
    tfidf = TfidfVectorizer()   #Make TFID vectorizer
    
    convAsStringList = []
    #Convert recommendation portion of conversations into a list of string
    for id in idList:  
        curString = " ".join(wholeConv[hash(id)][0])
        convAsStringList.append(curString)
    
    #Calculate TFIDF
    tfidfScores = tfidf.fit_transform(convAsStringList)
    realValues = tfidfScores.nonzero() #Ignore 0 values in the sparse matrix
    
    #Loop over each conversation and add up word importance 
    for docIdx in set(realValues[0]):         
        #Sum up each word in the document as the word importance score
        # so this is cool, use docidx for location, then sum across all the columns with slicing. 
        sumtfidf = tfidfScores[docIdx, :].sum()
        scoresList.append(sumtfidf)       
    return scoresList

#Sanity check that the tfidf vectorizer is working on the same conversations    
def verifyDocument(idList, wholeConv, docIndex=0):
    tfidf = TfidfVectorizer()
    
    #Convert recommendation portion of conversations into a list of strings
    curString = " ".join(wholeConv[hash(idList[docIndex])][0])

    #Calculate TFIDF
    tfidfScores = tfidf.fit_transform([curString])

    #Get the feature names from the TF-IDF vectorizer
    featureNames = tfidf.get_feature_names_out()

    # Get the nonzero and feature names for the document
    nonZeroIndices = tfidfScores.nonzero()
    nonZeroFeatureNames = [featureNames[idx] for idx in nonZeroIndices[1]] #This one is cool too

    # Print the original conversation and the feature names from TF-IDF
    print("Original Conversation:")
    print(curString)
    
    print("\nFeature Names from TF-IDF:")
    print(nonZeroFeatureNames)

#End word importance functions 
##########################################################################################

##########################################################################################
#Repitition functions
def scoreRepitition(idList, wholeConv):
    repitionScores = []
    #Loop over conversations
    for id in idList:        
        repeatedWords = 0
        curString = " ".join(wholeConv[hash(id)][0])
        
        #remove stop words
        tokenizedString = nltk.word_tokenize(curString)
        setString = set(tokenizedString)
        if STOP_WORDS.intersection(setString):
            setString -= STOP_WORDS
        
        #Search for repeated words
        for word in setString:
            if tokenizedString.count(word) > 1:
                repeatedWords +=1
        repitionScores.append(repeatedWords)
    return repitionScores
    
#End repitition function
##########################################################################################

##########################################################################################
#Subjectivity functions
def scoreSubjectivity(idList, wholeConv):
    subjectivityScores = []
    for id in idList:                        
        curString = " ".join(wholeConv[hash(id)][0])
        blob = TextBlob(curString)
        subjectivityScores.append(blob.sentiment.subjectivity)
    
    return subjectivityScores

#End subjectivity functions 
##########################################################################################

##########################################################################################
#Polarity functions
def scorePolarity(idList, wholeConv):
    polarityScores = []
    for id in idList:                        
        curString = " ".join(wholeConv[hash(id)][0])
        blob = TextBlob(curString)
        polarityScores.append(blob.sentiment.polarity)

    return polarityScores

#End Polarity functions 
##########################################################################################

##########################################################################################
#Grammitical correctness functions
def scoreGrammar(idList, wholeConv, startIndex,stopIndex, movies): #idList, wholeConv):

    '''    grammarScores = []
    for id in idList:                       
        curString = " ".join(wholeConv[hash(id)][0])
        tokens = nltk.word_tokenize(curString)
        #Get rid of 2nd portion of contractions in token list to avoid being penalized for how tokenization works
        cleanedTokens = [word for word in tokens if word[0] != '\'']
        grammarScores.append(spellCheckSentence(cleanedTokens))

            
    return grammarScores '''

    global grammarDict
    for i in range(startIndex, stopIndex): #len(idList)
        id = idList[i]
        curString = " ".join(wholeConv[hash(id)][0])

        #Remove any movies from INSPIRED that are in the sentence as those do not come with special notataion that can be parsed out. 
        #Do this to avoid counting movie titles which may have unique or nonsensical words as spelling errors
        #Movies in eredial come bracketed in [] and are avoided in spellchecksentence
        if hash(id) in movies.keys():
            for movieName in movies[hash(id)]:
                if movieName in curString:
                    curString = curString.replace(movieName, '')

    
        tokens = nltk.word_tokenize(curString)
        #Get rid of 2nd portion of contractions in token list to avoid being penalized for how tokenization works
        cleanedTokens = [word for word in tokens if word[0] != '\'']
        grammarDict[hash(id)] = spellCheckSentence(cleanedTokens)


def spellCheckSentence(tokenList):

    #Initialize spell checker
    spell = SpellChecker()
    conv_length = len(tokenList)

    #Flag to skip spellchecking movie titles which may have abnormal spelling/conventions
    skipCheck = False 
    errNo = 0
    for word in tokenList:
        word.replace('.','') 
        
        #Movies are surrounded by brackets in the e redial dataset.   
        #Skip spell check for Movie titles as they may have strange words / spellings
        if word == '[':
            skipCheck = True
        elif word == ']': #Movie title is over add in title then year
            skipCheck = False    
        elif word not in STOP_WORDS and word not in string.punctuation:
            ogWord = word
            word = spell.correction(word)
            if ((word is not None) and (ogWord != word)):
                errNo+=1

    #Score determined by number of errors / total setntence length
    score = errNo / conv_length
    return score


#Use threads to speed up grammar checking 
def threadProcessing(idList, wholeConv,movies):
    
    
  
    #Proces the dataset with some threads. Start with conversation turn list
    threadList = []
    numTasks = len(idList)//NUMTHREADS
    leftovers = len(idList)%NUMTHREADS
    startNum = 0
    endNum = numTasks
    for i in range(NUMTHREADS):

        #Assign any extra work to the last thread ALWAYS
        if(endNum == len(idList)-leftovers):
            endNum+=leftovers
        thread = threading.Thread(target=scoreGrammar, args = (idList, wholeConv, startNum, endNum,movies))
        threadList.append(thread)
        thread.start()
        #Adjust start and end numbers for next thread
        startNum+=numTasks
        endNum += numTasks
        

    #Wait for threads to finish
    for thread in threadList:
        thread.join()
    
    
    #Convert grammar dictionary to a list so it matches the format of the other score functions
    gramScores = []
    for key in grammarDict.keys():
        gramScores.append(grammarDict[key])


    return gramScores
#END grammatical correctness functions
##########################################################################################

##########################################################################################
#Feature Appearance functions
#This function will summarize a conversation using Bart, and calculate the cosine similarity using Bert
#It will then get the cosine simiarity of the embeddings to measure feature appearance
def scoreFeatureAppearance(idList, wholeConv):

    #Loop through every conversation in the list
    targetScores = []
    count = 0
    for id in idList:
        count+=1
        print('Working on id#',count)
        #Get current conversation, split into recommender and seeker (left/right)
        curConv = wholeConv[hash(id)]
        leftString = ' '.join(curConv[0])
        rightString = ' '.join(curConv[1])

        #Make summary of the conversation
        leftSummary = generateSummaryBart(leftString)
        rightSummary = generateSummaryBart(rightString)

        #Calculate the embedding of the summary
        leftEmbedding = calcBertEmbeddings(leftSummary)
        rightEmbedding = calcBertEmbeddings(rightSummary)

        #Calculate the cosine similarity of the 2 summaries
        similarity = cosine_similarity(leftEmbedding, rightEmbedding).item()
        targetScores.append(similarity)

    #return target scores
    return targetScores

#This function generate summaries using BART. 
def generateSummaryBart(text):
    inputs = tokenizer_bart(text, return_tensors="pt", max_length=1024, truncation=True)
    summary_ids = model_bart.generate(**inputs, max_length=150, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer_bart.decode(summary_ids[0], skip_special_tokens=True)
    return summary

#This  Function calculates the BERT embeddings for entire summarized text
#Apparently BART is better at summarizing but BERT is better for embeddings So we're mixing the 2. IDK this is where we are at right now...
def calcBertEmbeddings(text):
    tokens = tokenizer_bert(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        outputs = model_bert(**tokens)
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings


#End feature appearance functions
##########################################################################################
#Get any relevant data from eredial
def collectEredialData(filename):
    #Data format
    #messages
    #text
    #movies
    #knowledge
    #role
    #rewrite
    #reason
    #description
    #encourage
    #plot
    #review
    #movieid
    #wiki
    #conversationID


    #Open up our file
    with open(filename, 'r', encoding='utf-8') as f:
        messages = json.load(f)

    #Data already in a list. SO when you loop through it, it is going
    #Through a dictionary of every conversation#Each conversation
    #has 2 main parts messages and conversationId
    #Messages is the supermajority and has many subcategories and will need
    #the most processing. 

    #Containers for data            #Dict Contents
                                    #Key = conversationID

    #Will be used to evaluate each conversation as a whole
    wholeConv = {}                  #{[helperText],[HelperText]}

    #Will be used to evaluate each turn of dialogue
    #A turn being defined as each person having their full say
    #IE Person A says what they want, Person B then responds
    #Once person A speaks again, the turn is over. So a turn may be 
    #Defined as A*B*
    turnDict = {}                   #{[[helper],[seeker]]}

    #Will hold all known movie details
    moviesDict = {}                 #[wiki, plot, review, title]
    #Will hold Conv IDs
    idList = []                     #[conv ID]
    
    
    preservedConversation = {}         #Late addition added to ensure order is preserved to proper position embeddings can be added for pretrained model tokenize and stuff
    convRolesList = []

    #This will loop through each entry, 1 conversation at a time
    for message in messages:

        #In conjunction, these will record everything that 
        #has been said in a conversation
        totalHelperMssg = []
        totalSeekerMssg = []
        curRoles = []
        
        
        preservedOrderList = [] #Will hold conv exactly as it occurs

        #Will hold the turns of a conversation
        turnList = []      #[[[seeker],[helper]]]
        seekerTurnList = []
        helperTurnList = []

        #Will hold Movies
        movieList = []
        #track how many movies mentioned in a conv [for indexing]. 
        movieNum = -1
        #record all movie data
        fullMovieData = []
        

        #Will manage when a turn has occured
        seekerSpoken = False
        helperSpoken = False
                
        #Gather the set of messages in the conversation
        workin = message['messages']

        #Get conv ID and add to list
        id = message['conversationId']
        idList.append(id)
        
        firstSpeaker = workin[0]['role']

        #Loop through a conversation 
        for dictionary in workin:
            #Get current sentence    
            sentence = dictionary['text']
            #Determine who is speaking
            role = dictionary['role']
            if role == 1:
                textRole = "RECOMMENDER"
            else:
                textRole = "SEEKER"
            curRoles.append(role)
            preservedOrderList.append([textRole,sentence])
            
            #Manage conversation turn tracking 
            if role == 1:
                totalHelperMssg.append(sentence)
                helperTurnList.append(sentence)
                helperSpoken = True
                                
            else:
                totalSeekerMssg.append(sentence)
                seekerTurnList.append(sentence)
                seekerSpoken = True
                                
            #If both have spoken and a speaker is going again
            #record this turn of the conversation
            if helperSpoken and seekerSpoken and role == firstSpeaker:

                #Add this turn to a list
                turnList.append([helperTurnList,seekerTurnList])

                #Reset turn trackers
                seekerSpoken = False
                helperSpoken = False

                #Reset turn recorder lists
                seekerTurnList = []
                helperTurnList = []
                

            if 'movies' in dictionary.keys() and len(dictionary['movies']) != 0:
                #Remove [] from around the movie title
                for movie in dictionary['movies']:
                    cleanTitle = re.sub("[\\[\\]]", '', movie)
                    
                    movieList.append(cleanTitle)  
                    movieNum +=1  
                        
            if 'knowledge' in dictionary.keys() and len(dictionary['knowledge']) != 0:
                #Subparts to knowledge
                #wiki,plot,review,movieid

                wiki = None
                plot = None
                review = None

                #Get the knowledge dictionary
                
                for i in range(len(dictionary['knowledge'])):
                    
                    curKnowledge = dictionary['knowledge'][i]
                    
                    wiki = curKnowledge['wiki']
                    plot = curKnowledge['plot']
                    review = curKnowledge['review']

                    #Remove extra newlines from the sentences
                    if wiki is not None:
                        wiki = re.sub(r'[\r\n]', ' ', wiki)
                    if plot is not None:
                        plot = re.sub(r'[\r\n]', ' ', plot)
                    if review is not None:    
                        review = re.sub(r'[\r\n]', ' ', review)
                    
                    fullMovieData.append([wiki, plot, review])
                    
                    #description
                    #encourage

        #Add info to conv containers
        wholeConv[hash(id)] = [totalHelperMssg, totalSeekerMssg]
        turnDict[hash(id)] = turnList        
        #Add movie name to movie data
        for i in range(len(movieList)):
            fullMovieData[i].append(movieList[i])
        #Store full conv data about movies
        moviesDict[hash(id)] = fullMovieData
        convRolesList.append(curRoles)
        #Preserve the order in which the entire conversation was spoken
        preservedConversation[hash(id)] = preservedOrderList

    #add on whatever else as necessary
    return idList, wholeConv, preservedConversation

#Get data from inspired dataset    
def collectInspiredData(Inspired_path, idList, wholeConv,preservedConversation):
    seeker_list = []
    recommender_list = []
    movies = {}
    with open(Inspired_path, "r", encoding='utf-8') as file:
        csv_reader = csv.reader(file, delimiter='\t')
        next(csv_reader)
        

        last_role = ""
        last_convid = ""
        dialog = []
        dialog_list = collections.OrderedDict()
        preservedOrderList = []
            
        seeker_intend_info = collections.OrderedDict({"movie":[], "genre":[], "people_name":[]})
        recommender_intend_info = collections.OrderedDict({"movie":[], "genre":[], "people_name":[]})

        for row in csv_reader:
            conv_idx = row[0]
            
            if conv_idx not in idList:
                idList.append(conv_idx)
            role = row[2]
            utterance = row[5]
            strategy = row[14]
            preservedOrderList.append([role,row[4]])
            
            movies[hash(conv_idx)] = row[9]
            if (last_convid != "") and (last_convid != conv_idx):
                wholeConv[hash(last_convid)] = [recommender_list,seeker_list]
                preservedConversation[hash(last_convid)] = preservedOrderList
                preservedOrderList= []
                recommender_list = []
                seeker_list = []
                
            
            
            if strategy == "transparency":
                strategy = "<offer_help> "
            else:
                strategy = "<" + strategy +">"
            if strategy =="<>":
                strategy = ""
            
    #         seeker intention info
            if role == "SEEKER":
                seeker_list.append(row[4])
                if row[6] != "":
                    seeker_intend_info["movie"].append(row[6].replace(";", ",").replace("  ", " "))
                    
                if row[7] != "":
                    seeker_intend_info["genre"].append(row[7].replace(";", ",").replace("  ", " "))
                if row[8] != "":
                    seeker_intend_info["people_name"].append(row[8].replace(";", ",").replace("  ", " "))
            
            elif last_role == "SEEKER":
                if len(seeker_intend_info["movie"]) != 0:
                    movie_info = "movie: "+", ".join(seeker_intend_info["movie"])+";"
                else: 
                    movie_info = ""
                    
                if len(seeker_intend_info["genre"]) != 0:
                    genre_info = "genre: "+ ", ".join(seeker_intend_info["genre"]) +";"
                else:
                    genre_info = ""
                
                if len(seeker_intend_info["people_name"]) != 0:
                    people_info = "people_name: " + ", ".join(seeker_intend_info["people_name"])+";"
                else:
                    people_info = ""
                    
                # previous seeker may be in the same conversation or in the last conversation
                if conv_idx == last_convid:
                    index = conv_idx
                else:
                    index = last_convid
                if (len(seeker_intend_info["movie"]) != 0) or (len(seeker_intend_info["genre"]) != 0) or (len(seeker_intend_info["people_name"]) != 0):
                    dialog_list[index][-1] = dialog_list[index][-1] + " [SEP]"
                    
                    if movie_info !=0:   
                        dialog_list[index][-1] = dialog_list[index][-1] + movie_info
                    if genre_info != 0:
                        dialog_list[index][-1] = dialog_list[index][-1] + genre_info
                    if people_info != 0:
                        dialog_list[index][-1] = dialog_list[index][-1] + people_info
                seeker_intend_info = collections.OrderedDict({"movie":[], "genre":[], "people_name":[]})

    #         seeker intention info
            if role == "RECOMMENDER":
                recommender_list.append(row[4])
                if row[6] != "":
                    recommender_intend_info["movie"].append(row[6].replace(";", ",").replace("  ", " "))
                    
                if row[7] != "":
                    recommender_intend_info["genre"].append(row[7].replace(";", ",").replace("  ", " "))
                if row[8] != "":
                    recommender_intend_info["people_name"].append(row[8].replace(";", ",").replace("  ", " "))
            
            elif last_role == "RECOMMENDER":
                if len(recommender_intend_info["movie"]) != 0:
                    movie_info = "movie: "+", ".join(recommender_intend_info["movie"])+";"
                else: 
                    movie_info = ""
                    
                if len(recommender_intend_info["genre"]) != 0:
                    genre_info = "genre: "+ ", ".join(recommender_intend_info["genre"]) +";"
                else:
                    genre_info = ""
                
                if len(recommender_intend_info["people_name"]) != 0:
                    people_info = "people_name: " + ", ".join(recommender_intend_info["people_name"])+";"
                else:
                    people_info = ""
                    
                # previous seeker may be in the same conversation or in the last conversation
                if conv_idx == last_convid:
                    index = conv_idx
                else:
                    index = last_convid
                if (len(recommender_intend_info["movie"]) != 0) or (len(recommender_intend_info["genre"]) != 0) or (len(recommender_intend_info["people_name"]) != 0):
                    dialog_list[index][-1] = dialog_list[index][-1] + " [SEP]"
                    
                    if movie_info !=0:   
                        dialog_list[index][-1] = dialog_list[index][-1] + movie_info
                    if genre_info != 0:
                        dialog_list[index][-1] = dialog_list[index][-1] + genre_info
                    if people_info != 0:
                        dialog_list[index][-1] = dialog_list[index][-1] + people_info
                recommender_intend_info = collections.OrderedDict({"movie":[], "genre":[], "people_name":[]})

            
            if conv_idx in dialog_list.keys():
                if role == last_role:
                    dialog_list[conv_idx][-1] = dialog_list[conv_idx][-1] + " " + strategy + utterance
                else:
                    dialog_list[conv_idx].append(role +":" + strategy + utterance)
            else:
                dialog_list[conv_idx] = [role + ":" + utterance]
                
            last_role = role

            

            last_convid = conv_idx
            
        #Add last entry
        wholeConv[hash(last_convid)] = [recommender_list,seeker_list]
        preservedConversation[hash(last_convid)] = preservedOrderList
        
        #Convert movies to just movie names, removing year markers
        for key in movies.keys():
            #When taking row data it is read in as a string. Use AST to conver string to dictionary for processing
            temp_dict = ast.literal_eval(movies[key])
            #Movie names are the keys to the dictionary, grab the movie name
            temp_list = [key for key in temp_dict.keys()]
            #Split each movie on the opening parenthesis and take just the 1st bit since that is the title. Ignore the year it was released
            temp_list = [title.split(' (')[0] for title in temp_list]
            movies[key] = temp_list
        
        return idList, wholeConv, movies, preservedConversation

#Takes in a dictionary of conversations, and the idlist which is the keys to that dictionary
#Goes through and combines each conversation so that any consecutive messages from a speaker are 
#joined into one entry, that way each conversation has the form speaker 1, speaker 2 speaker 1 for consistency
def combineConsecutiveSpeakerSentences(idList, preservedConversation):
    joined_strings_list = []

    for id, conv in zip(idList, preservedConversation):
        current = preservedConversation[hash(id)]
        joined_strings = []
        current_role = None
        current_string = ""

        for role, text in current:
            if current_role is None:
                # First iteration
                current_role = role
                current_string = text
            elif current_role == role:
                # Same role, concatenate the strings
                current_string += " " + text
            else:
                # Different role, start a new string
                joined_strings.append(current_role + ': ' +current_string)
                current_role = role
                current_string = text

        # Append the last string
        joined_strings.append(current_string)
        joined_strings_list.append(joined_strings)

    return joined_strings_list    
#Takes in a list of conversation IDs, and a dictionary of conversations where the key is in the parallel IDlist. Also accepts lists of score values that are all parallel arrays.
#Will write the data to the file in csv format as: id,seekerconv,recommenderConv,length score,readabilityscores, word importance score, repitition score, 
#subjectivity score, polarity score, grammar score, featuer appearance score
    
def writeCombinedData(idList,wholeConv, lenScores, readScores, wiScores,repScores,subScores,polScores,gramScores,faScores,preservedConversation, fileChoice):
    if fileChoice == 1:
        outFile = open('TRAIN_combinedData.txt','w', encoding='utf-8')
    else:
        outFile = open('TEST_combinedData.txt','w', encoding='utf-8')
    for x, id in enumerate(idList):
        outFile.write(f'{id}|={wholeConv[hash(id)][1]}|={wholeConv[hash(id)][0]}|={lenScores[x]}|={readScores[x]}|={wiScores[x]}|={repScores[x]}|={subScores[x]}|={polScores[x]}|={gramScores[x]}|={faScores[x]}|={preservedConversation[x]}\n') 
    outFile.close()

#

######################################################################################################################################################
#Stat crunching section. Functions to gather statistic on the quality indicators for reporting. 
#Takes in score lists for each of the quality indicators which are all parallel arrays. 
#Will calculate the min,max,mean,median,range, stddev of each list, and store in a dictionary of dictionaries for each value
def crunchStats(lenScores,readScores,wiScores,repScores,subScores,polScores,gramScores, faScores):
    dataDict = {}
    superList = [lenScores,readScores,wiScores,repScores,subScores,polScores,gramScores, faScores]
    keyNames = ["length","readability","wordimportance","repetition","subjectivity","polarity","grammar","featureappearance"]
    for x,list in enumerate(superList):
        curDict = {}
        minVal = min(list)
        maxVal = max(list)
        average = sum(list) / len(list)
        median = getMedian(list)
        range = maxVal - minVal
        stdDev = getStdDev(list, average)
        curDict["min"] = minVal
        curDict["max"] = maxVal
        curDict["mean"] = average
        curDict["median"] = median
        curDict["range"] = range
        curDict["stddev"] = stdDev
        dataDict[keyNames[x]] = curDict
    

    return dataDict

def getMedian(data):
    sortedData = sorted(data)
    middle = len(sortedData)//2
    if len(sortedData) %2 ==0:
        median = (sortedData[middle-1] + sortedData[middle]) /2
    else:
        median = sortedData[middle]

    return median 

def getStdDev(data, average):
    stdDev = 0
    for val in data:
        stdDev += (val-average)**2 

    stdDev=  (stdDev/len(data))**.5
    return stdDev

#Will write the stats from crunch stats to a file
def writeQFStats(dataDict, fileChoice):

    if fileChoice == 1:
        outFile = open("TRAIN_QualityFactorStats.csv","w", encoding='utf-8')
    else:
        outFile = open("TEST_QualityFactorStats.csv","w", encoding='utf-8')

    #Write csv column names as 1st line
    outFile.write("FactorName,min,max,mean,median,range,stddev\n")
    #Loop through each quality factor in the data dictionary
    for qualityKey in dataDict.keys():
        
        #Get current working dictionary
        curDict = dataDict[qualityKey]
        
        #Write the name of the quality factor as line heading
        outFile.write(f'{qualityKey},')
        
        #Loop through each stat in the subDictionary and record on each line
        #Substats are in this order: min,max,mean,median,range,stddev
        for subKey in curDict.keys():
            outFile.write(f'{curDict[subKey]},')
        outFile.write('\n')

    outFile.close()
######################################################################################################################################################
#Call Main function
if __name__ == '__main__':
    main()