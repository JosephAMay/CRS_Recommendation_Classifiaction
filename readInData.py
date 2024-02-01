import matplotlib.pyplot as plt
import numpy as np
def main():

    keepGoing = True
    while keepGoing:
        choice = int(input("Enter 1 to read in training data, 2 to process test data: "))
        if choice ==1 or choice ==2:
            keepGoing =False
    idList,seekerConv,recommenderConv,length,readability,wordImp,repetition,subjectivity,polarity,grammar,featureAppearance = readInData(choice)
    scoreConv(idList,length,readability,wordImp,repetition,subjectivity,grammar,featureAppearance,seekerConv,recommenderConv,choice)      
    #showHistograms(idList,length,readability,wordImp,repetition,subjectivity,grammar,featureAppearance,seekerConv,recommenderConv)
        

#Open up the file, store the data with each column in its own array.
#Data in the file is delimited by |= 
def readInData(choice):
    if choice == 1:
        filename= 'C:\\Users\\josep\\OneDrive\\Desktop\\ThesisWork\\E-Redial\\dataset\\TRAIN_combinedData.csv'
    else:
        filename= 'C:\\Users\\josep\\OneDrive\\Desktop\\ThesisWork\\E-Redial\\dataset\\TEST_combinedData.csv'
    idList = []
    seekerConv = []
    tempSeekerConv = []
    recommenderConv = []
    tempRecommenderConv=[]
    length = []
    repetition = []
    readability = []
    wordImp = []
    grammar = []
    featureAppearance = []
    polarity = []
    subjectivity = []
    #id,seekerconv,recommenderConv,length score,readabilityscores, word importance score, repitition score, 
    #subjectivity score, polarity score, grammar score, featuer appearance score
    with open(filename, "r", encoding='utf-8') as file:
        row=[]
        for line in file:
            row = line.split('|=')
        
            idList.append(row[0])
            tempSeekerConv.append(row[1])
            tempRecommenderConv.append(row[2])
            length.append(float(row[3]))
            readability.append(float(row[4]))
            wordImp.append(float(row[5]))
            repetition.append(float(row[6]))
            subjectivity.append(float(row[7]))
            polarity.append(float(row[8]))
            grammar.append(float(row[9]))
            featureAppearance.append(float(row[10]))
   
    #Convert strings read in from file into true lists for seeker and recommender conversations
    for j , item in enumerate(tempSeekerConv):
        convertedList = eval(tempSeekerConv[j])
        seekerConv.append(convertedList)
    for j , item in enumerate(tempRecommenderConv):
        convertedList = eval(tempRecommenderConv[j])
        recommenderConv.append(convertedList)
    return idList,seekerConv,recommenderConv,length,readability,wordImp,repetition,subjectivity,polarity,grammar,featureAppearance


    
    
#Score conversation based on breakpoints observed in data / having desirable conditions
def scoreConv(idList,length,readability,wordImp,repetition,subjectivity,grammar,featureAppearance,seekerConv,recommenderConv,choice):
    if choice == 1:
        filename= 'TRAIN_Labels_combinedData.csv'
    else:
        filename= 'TEST_Labels_combinedData.csv'


    outFile = open(filename, 'w', encoding ='utf-8')
    #Loop through data scores, based on score, assign a label
    for id,lenVal,readVal,wordVal,repVal,subVal,gramVal,feaVal in zip(idList,length,readability,wordImp,repetition,subjectivity,grammar,featureAppearance):
        #Very good conversation: Label == 1
        if (lenVal >=.3) and (readVal >=7) and (wordVal >=7) and (repVal <=20) and (subVal >=.45) and (gramVal <=.04) and (feaVal >=.82):
            outFile.write(f'{id},{1}\n')
        #Average Conversation: Label == 2
        elif (lenVal >=.25) and (readVal >5) and (wordVal >4) and (repVal <=25) and (subVal >=.30) and (gramVal <=.05) and (feaVal>=.8):
            outFile.write(f'{id},{2}\n')
        #Bad conversation: Label == 3
        else:
            outFile.write(f'{id},{3}\n')
    
    outFile.close()



#Make some histograms of the data for quick visualization of the spread of the data. 
def showHistograms(idList,length,readability,wordImp,repetition,subjectivity,grammar,featureAppearance,seekerConv,recommenderConv):
    plt.hist(length, bins = 10, color='blue', edgecolor='black')
    plt.xlabel('Value')
    plt.title("Length")
    plt.ylabel('Frequency')
    plt.title('Histogram of length')
    # Calculate the bin edges and add labels
    bin_edges, _ = np.histogram(length, bins=10)
    for i in range(len(bin_edges) - 1):
        start = bin_edges[i]
        end = bin_edges[i + 1]
        plt.text((start + end) / 2, 0, f'{start:.2f}-{end:.2f}', ha='center', va='bottom')


    # Display the plot
    plt.show()

    plt.hist(repetition, bins = 10, color='blue', edgecolor='black')
    plt.xlabel('Value')
    plt.title("Repetition")
    plt.ylabel('Frequency')
    plt.title('Histogram of repetition')
    # Calculate the bin edges and add labels
    bin_edges, _ = np.histogram(length, bins=10)
    for i in range(len(bin_edges) - 1):
        start = bin_edges[i]
        end = bin_edges[i + 1]
        plt.text((start + end) / 2, 0, f'{start:.2f}-{end:.2f}', ha='center', va='bottom')


    # Display the plot
    plt.show()

    plt.hist(readability, bins = 10, color='blue', edgecolor='black')
    plt.xlabel('Value')
    plt.title("Readability")
    plt.ylabel('Frequency')
    plt.title('Histogram of readability')
    # Calculate the bin edges and add labels
    bin_edges, _ = np.histogram(length, bins=10)
    for i in range(len(bin_edges) - 1):
        start = bin_edges[i]
        end = bin_edges[i + 1]
        plt.text((start + end) / 2, 0, f'{start:.2f}-{end:.2f}', ha='center', va='bottom')


    # Display the plot
    plt.show()

    plt.hist(polarity, bins = 10, color='blue', edgecolor='black')
    plt.xlabel('Value')
    plt.title("Polarity")
    plt.ylabel('Frequency')
    plt.title('Histogram of polarity')
    # Calculate the bin edges and add labels
    bin_edges, _ = np.histogram(length, bins=10)
    for i in range(len(bin_edges) - 1):
        start = bin_edges[i]
        end = bin_edges[i + 1]
        plt.text((start + end) / 2, 0, f'{start:.2f}-{end:.2f}', ha='center', va='bottom')


    # Display the plot
    plt.show()

    plt.hist(subjectivity, bins = 10, color='blue', edgecolor='black')
    plt.xlabel('Value')
    plt.title("Subjectivity")
    plt.ylabel('Frequency')
    plt.title('Histogram of subjectivity')
    # Calculate the bin edges and add labels
    bin_edges, _ = np.histogram(length, bins=10)
    for i in range(len(bin_edges) - 1):
        start = bin_edges[i]
        end = bin_edges[i + 1]
        plt.text((start + end) / 2, 0, f'{start:.2f}-{end:.2f}', ha='center', va='bottom')


    # Display the plot
    plt.show()

    plt.hist(grammar, bins = 10, color='blue', edgecolor='black')
    plt.xlabel('Value')
    plt.title("Grammar")
    plt.ylabel('Frequency')
    plt.title('Histogram of grammar')
    # Calculate the bin edges and add labels
    bin_edges, _ = np.histogram(length, bins=10)
    for i in range(len(bin_edges) - 1):
        start = bin_edges[i]
        end = bin_edges[i + 1]
        plt.text((start + end) / 2, 0, f'{start:.2f}-{end:.2f}', ha='center', va='bottom')


    # Display the plot
    plt.show()

    plt.hist(featureAppearance, bins = 10, color='blue', edgecolor='black')
    plt.xlabel('Value')
    plt.title("Feature Appearnce")
    plt.ylabel('Frequency')
    plt.title('Histogram of feature apperance')
    # Calculate the bin edges and add labels
    bin_edges, _ = np.histogram(length, bins=10)
    for i in range(len(bin_edges) - 1):
        start = bin_edges[i]
        end = bin_edges[i + 1]
        plt.text((start + end) / 2, 0, f'{start:.2f}-{end:.2f}', ha='center', va='bottom')


    # Display the plot
    plt.show()

    plt.hist(wordImp, bins = 10, color='blue', edgecolor='black')
    plt.xlabel('Value')
    plt.title("Word Importance")
    plt.ylabel('Frequency')
    plt.title('Histogram of wordImportance')
    # Calculate the bin edges and add labels
    bin_edges, _ = np.histogram(length, bins=10)
    for i in range(len(bin_edges) - 1):
        start = bin_edges[i]
        end = bin_edges[i + 1]
        plt.text((start + end) / 2, 0, f'{start:.2f}-{end:.2f}', ha='center', va='bottom')


    # Display the plot
    plt.show()
    

            
            



main()