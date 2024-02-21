#!/usr/bin/env python3
import numpy as np
import torch
import pytorch_lightning as pl
import torchmetrics
import torchvision
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer, GPT2Model
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
import pickle
import random
# Train data 1557 long. 1557 / 9 == 173  even runs through data.
#Test data 249 long. 249 / 3 == 83 even runs through data.
TRAIN_BATCH_SIZE = 9
TEST_BATCH_SIZE = 3
NUM_CLASSES=3
NUM_EPOCHS = 80



# Load pre-trained GPT-2 model and tokenizer
model = GPT2Model.from_pretrained('gpt2')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer.add_special_tokens({'pad_token': '[PAD]'})


# Check system requirements
if torch.cuda.is_available():
    print(torch.cuda.get_device_name())
    print(torch.cuda.get_device_properties("cuda"))
    print("Number of devices:", torch.cuda.device_count())
    device = ("cuda")
else:
    print("Only CPU is available...")
    device = ("cpu")

#Open up the file, store the data with each column in its own array.
#Data in the file is delimited by |= 
def readInData(choice):
    if choice == 1 or choice =='training':
        filename= 'Normalized_TRAIN_combinedData.txt'
    else:
        filename= 'Normalized_TEST_combinedData.txt'
    idList = []
    seekerConv = []
    tempSeekerConv = []
    recommenderConv = []
    tempRecommenderConv=[]
    tempPreservedOrder = []
    preservedOrder = []
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
            tempPreservedOrder.append(row[11])
   
    #Convert strings read in from file into true lists for seeker and recommender conversations
    for j , item in enumerate(tempSeekerConv):
        convertedList = eval(tempSeekerConv[j])
        seekerConv.append(convertedList)
    for j , item in enumerate(tempRecommenderConv):
        convertedList = eval(tempRecommenderConv[j])
        recommenderConv.append(convertedList)
    for j, item in enumerate(tempPreservedOrder):
        convertedList = eval(tempPreservedOrder[j])
        preservedOrder.append(convertedList)
    return idList,seekerConv,recommenderConv,length,readability,wordImp,repetition,subjectivity,polarity,grammar,featureAppearance, preservedOrder

#Reads in label Data. Data comes as ConvID,Label
def readInLabels(choice):
    if choice == 1 or choice =='training':
        filename='TRAIN_Labels_combinedData.csv'
    else:
        filename='TEST_Labels_combinedData.csv'

    labels = []
    with open(filename, "r", encoding='utf-8') as file:
        for line in file:
            cur = line.split(',')
            labels.append(int(cur[1])) #Label is the 2nd element on the line. Ignore convID

    return labels

# Function to pad conversations to a common length
def padConversation(conversations):
    # Get maxlength conversation, and pad each conversation so they're the same length
    max_length = max(len(conv) for conv in conversations)
    padded_conversations = []
    for conv in conversations:
        padded_conv = conv + [' '] * (max_length - len(conv))
        padded_conversations.append(padded_conv)
    return padded_conversations
    

class encoderNetwork(Dataset):
    def __init__(self, tokenizer, preserved_order, length,readability,wordImp,repetition,subjectivity,polarity,grammar,featureAppearance, labels):
        self.tokenizer = tokenizer
        self.conversations = preserved_order
        self.length = length
        self.readability = readability
        self.wordImp = wordImp
        self.repetition = repetition
        self.polarity = polarity
        self.subjectivity = subjectivity
        self.featApp = featureAppearance
        self.grammar = grammar
        self.labels = labels

    def __len__(self):
        return len(self.conversations)

    def __getitem__(self, idx):

        text = " ".join(self.conversations[idx])
        encoding = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")

        input_ids = encoding["input_ids"]
        attention_mask = encoding["attention_mask"]
        
        input_ids = encoding["input_ids"].clone().detach().to(device)
        attention_mask = encoding["attention_mask"].clone().detach().to(device)
        lenScore = torch.tensor(self.length[idx]).view(-1, 1).clone().detach().to(device)
        readScore = torch.tensor(self.readability[idx]).view(-1, 1).clone().detach().to(device)
        wordScore = torch.tensor(self.wordImp[idx]).view(-1, 1).clone().detach().to(device)
        repScore = torch.tensor(self.repetition[idx]).view(-1, 1).clone().detach().to(device)
        polScore =  torch.tensor(self.polarity[idx]).view(-1, 1).clone().detach().to(device)
        subScore =  torch.tensor(self.subjectivity[idx]).view(-1, 1).clone().detach().to(device)
        featScore =  torch.tensor(self.featApp[idx]).view(-1, 1).clone().detach().to(device)
        gramScore= torch.tensor(self.grammar[idx]).view(-1, 1).clone().detach().to(device)
        label = torch.tensor([int(self.labels[idx])], dtype=torch.long).clone().detach().to(device)       

        return input_ids, attention_mask, lenScore,readScore,wordScore,repScore,polScore,subScore,featScore,gramScore, label

    def __setitem__(self, idx,value):
        input_ids, attention_mask, length, readability, wordImp, repetition, subjectivity, polarity, grammar, featureAppearance, labels = value
        # Update the attributes at the given index with the new value
        self.length[idx] = length
        self.readability[idx] = readability
        self.wordImp[idx] = wordImp
        self.repetition[idx] = repetition
        self.subjectivity[idx] = subjectivity
        self.polarity[idx] = polarity
        self.grammar[idx] = grammar
        self.featApp[idx] = featureAppearance
        self.labels[idx] = labels

# Classify the conversation
class ClassifierNetwork(pl.LightningModule):
    def __init__(self, model, num_classes=NUM_CLASSES):
        super(ClassifierNetwork, self).__init__()

        # Freeze GPT-2 weights
        for param in model.parameters():
            param.requires_grad = False

        self.gpt2 = model

        #Projection layer to be used to add residual connections.
        self.projection = nn.Linear(NUM_CLASSES,256)
        self.residShrink = nn.Linear(256, NUM_CLASSES)
        #Linear Layers. Stacked into sets of 3
        self.fc1 = nn.Linear(1536, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, NUM_CLASSES)
        #Middle layer, projected down, then back up
        self.fc4 = nn.Linear(NUM_CLASSES, 256)
        self.fc5 = nn.Linear(256, 128)
        self.fc6 = nn.Linear(128, NUM_CLASSES)
        #Last layer, project back down again
        self.fc7 = nn.Linear(NUM_CLASSES, 256)
        self.fc8 = nn.Linear(256, 128)
        self.fc9 = nn.Linear(128, NUM_CLASSES)
        self.final = nn.Linear(256, NUM_CLASSES)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.classification.Accuracy(num_classes=NUM_CLASSES, task='multiclass')


    def forward(self, input_ids, attention_mask, lenScore,readScore,wordScore,repScore,polScore,subScore,featScore,gramScore):
        outputs = self.gpt2(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs['last_hidden_state']  # Use last hidden states instead of pooler_output

        # Take the mean of last hidden states along the sequence dimension
        pooled_output = torch.mean(last_hidden_states, dim=1)
        
        
        scores = torch.cat((lenScore, readScore, wordScore, repScore, polScore, subScore, featScore, gramScore), dim=1)
        repeated_scores = scores.repeat(1, pooled_output.size(1)//scores.size(1))
        # Combine pooled output and repeated scores along a new dimension
        combined_outputs = torch.cat((pooled_output, repeated_scores), dim=1)
        
        #linear layers with residual connections
        x = F.relu(self.fc1(combined_outputs))
        resid = x
        x = F.relu(self.fc2(x))
        x = F.relu(self.projection(self.fc3(x)) + resid)  # Residual connection from fc1 to fc3
        x = F.relu(self.residShrink(x))

        x = F.relu(self.fc4(x))
        resid = x
        x = F.relu(self.fc5(x))
        x = F.relu(self.projection(self.fc6(x)) + resid)  # Residual connection from fc4 to fc6
        x = F.relu(self.residShrink(x))

        x = F.relu(self.fc7(x))
        resid = x
        x = F.relu(self.fc8(x))
        x = F.relu(self.projection(self.fc9(x)) + resid)  # Residual connection from fc7 to fc9

        
        logits = self.final(x)

        return logits

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, lenScore, readScore, wordScore, repScore, polScore, subScore, featScore, gramScore, labels = batch
        logits = self(input_ids, attention_mask, lenScore, readScore, wordScore, repScore, polScore, subScore, featScore, gramScore)
        loss = self.loss_fn(logits, labels)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, labels)

        # Log metrics
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, lenScore, readScore, wordScore, repScore, polScore, subScore, featScore, gramScore, labels = batch
        logits = self(input_ids, attention_mask, lenScore, readScore, wordScore, repScore, polScore, subScore, featScore, gramScore)
        loss = self.loss_fn(logits, labels)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, labels)

        # Log metrics
        self.log('val_acc', acc, on_step=False, on_epoch=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=2e-5)

    def predict(self,batch):
        input_ids, attention_mask, lenScore, readScore, wordScore, repScore, polScore, subScore, featScore, gramScore, labels = batch
        logits = self(input_ids, attention_mask, lenScore, readScore, wordScore, repScore, polScore, subScore, featScore, gramScore)
        predicted_labels = torch.argmax(logits, dim=1)
        return predicted_labels
    
#Do shap nalysis 
def calcShap(dataset, backgroundData,model):
    totalPredictions = []
    for convIdx, conv in enumerate(dataset):
        convPrediction = []
        randidx = random.randint(0, len(backgroundData) - 1)
        for qidx in range(2,10): #start at length index, go up to 9 for featureAppearance
            
            prediction = model.predict(dataset[convIdx])
            
            
            #Grab factors of this conversation
            input_ids = dataset[convIdx][0]
            attention_masks = dataset[convIdx][1]
            length = dataset[convIdx][2]
            readability = dataset[convIdx][3]
            wordImp = dataset[convIdx][4]
            repetition = dataset[convIdx][5]
            polarity = dataset[convIdx][6]
            subjectivity = dataset[convIdx][7]
            featureAppearance = dataset[convIdx][8]
            grammar = dataset[convIdx][9]
            labels = dataset[convIdx][10]

            #grab current values to reset data back to original state after 
            #adjusting a vlue for shap
            oldlength = length
            oldreadability = readability
            oldwordImp = wordImp
            oldrepetition = repetition
            oldpolarity = polarity
            oldsubjectivity = subjectivity
            oldfeatureAppearance = featureAppearance
            oldgrammar = grammar
            
            if qidx == 2:
                length = backgroundData[randidx][2]
            elif qidx == 3:
                readability = backgroundData[randidx][3]
            elif qidx == 4:
                wordImp = backgroundData[randidx][4]
            elif qidx == 5:
                repetition = backgroundData[randidx][5]
            elif qidx == 6:
                polarity = backgroundData[randidx][6]
            elif qidx == 7:
                subjectivity = backgroundData[randidx][7]
            elif qidx == 8:
                featureAppearance = backgroundData[randidx][8]
            elif qidx == 9:
                grammar = backgroundData[randidx][9]

            dataset[convIdx] = input_ids,attention_masks,length,readability,wordImp,\
                    repetition,subjectivity,polarity,grammar,featureAppearance,labels
            
            backgroundPrediction = model.predict(dataset[convIdx])
            
            #Reset testdata to have original information
            dataset[convIdx] = input_ids,attention_masks,oldlength,oldreadability,oldwordImp,\
                    oldrepetition,oldsubjectivity,oldpolarity,oldgrammar,oldfeatureAppearance,labels
            shapValue = prediction-backgroundPrediction
            
            convPrediction.append(shapValue.item())
        totalPredictions.append(convPrediction)
    return np.array(totalPredictions)

def main():
    testidList,testSeekerConv,testRecommenderConv,testLength,testReadability,testWordImp,testRepetition,testSubjectivity,testPolarity,\
        testGrammar,testFeatureAppearance,testpreservedOrder = readInData('test')

    #Get label data for train / test
    testLabels = readInLabels('test')   
    #Pad input data so all conversations are the same length
    testInput = padConversation(testpreservedOrder)
    testDataset = encoderNetwork(tokenizer,testInput,testLength,testReadability,testWordImp,testRepetition,testSubjectivity,testPolarity,testGrammar,testFeatureAppearance,testLabels) 


    testidList = testidList[0:83]
    testSeekerConv = testSeekerConv[0:83]
    testRecommenderConv= testRecommenderConv[0:83]
    testLength =testLength[0:83]
    testReadability = testReadability[0:83]
    testWordImp = testWordImp[0:83]
    testRepetition = testRepetition[0:83]
    testSubjectivity = testSubjectivity[0:83]
    testPolarity = testPolarity[0:83]
    testGrammar =testGrammar[0:83]
    testFeatureAppearance = testFeatureAppearance[0:83]
    testpreservedOrder = testpreservedOrder[0:83]

    padConv = padConversation(testpreservedOrder)
    backgroundDataset = encoderNetwork(tokenizer,testInput,testLength,testReadability,testWordImp,testRepetition,testSubjectivity,testPolarity,testGrammar,testFeatureAppearance,testLabels) 

    #encoder classifier model
    classifier = ClassifierNetwork.load_from_checkpoint("gpt2_pretrain.ckpt",model=model, device=device)
    shapValues = calcShap(testDataset,backgroundDataset,classifier)
    np.savetxt('gpt2_shap.txt', shapValues)
if __name__ == "__main__":
    main()