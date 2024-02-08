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

# Train data 1557 long. 1557 / 9 == 173  even runs through data.
#Test data 249 long. 249 / 3 == 83 even runs through data.
TRAIN_BATCH_SIZE = 9
TEST_BATCH_SIZE = 3
NUM_CLASSES=3
NUM_EPOCHS = 80


# Load pre-trained GPT-2 model and tokenizer
model = GPTNeoModel.from_pretrained('EleutherAI/gpt-neo-125m')  # Update this line
tokenizer = GPT2Tokenizer.from_pretrained('EleutherAI/gpt-neo-125m')  # You can use GPT2Tokenizer as it's the same for GPT-Neo
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
        
        input_ids = encoding["input_ids"].clone().detach()
        attention_mask = encoding["attention_mask"].clone().detach()
        
        lenScore = torch.tensor(self.length[idx]).view(-1, 1).clone().detach()
        readScore = torch.tensor(self.readability[idx]).view(-1, 1).clone().detach()
        wordScore = torch.tensor(self.wordImp[idx]).view(-1, 1).clone().detach()
        repScore = torch.tensor(self.repetition[idx]).view(-1, 1).clone().detach()
        polScore =  torch.tensor(self.polarity[idx]).view(-1, 1).clone().detach()
        subScore =  torch.tensor(self.subjectivity[idx]).view(-1, 1).clone().detach()
        featScore =  torch.tensor(self.featApp[idx]).view(-1, 1).clone().detach()
        gramScore= torch.tensor(self.grammar[idx]).view(-1, 1).clone().detach()
        label = torch.tensor([int(self.labels[idx])], dtype=torch.long).clone().detach()
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

        self.gptneo = model

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
        
        outputs = self.gptneo(input_ids, attention_mask=attention_mask)
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

#Do ice analysis on given quality score        
def iceAnalysis(qidx,dataset,model):
    minVal=0
    maxVal=1
    steps = 101
    inc = (maxVal - minVal) / (steps - 1)
    totalPredictions = []
    for convIdx, conv in enumerate(dataset):
        convPrediction = []              
        for value in range(steps):
            iceVal=minVal+value*inc
            input_ids = dataset[convIdx][0]
            attention_masks = dataset[convIdx][1]
            length = dataset[convIdx][2]
            readability = dataset[convIdx][3]
            wordImp = dataset[convIdx][4]
            repetition = dataset[convIdx][5]
            subjectivity = dataset[convIdx][6]
            polarity = dataset[convIdx][7]
            grammar = dataset[convIdx][8]
            featureAppearance = dataset[convIdx][9]
            labels = dataset[convIdx][10]
            if qidx == 2:
                length = torch.tensor(iceVal, dtype=torch.float32).view(-1, 1)
            elif qidx == 3:
                readability = torch.tensor(iceVal, dtype=torch.float32).view(-1, 1)
            elif qidx == 4:
                wordImp = torch.tensor(iceVal, dtype=torch.float32).view(-1, 1)
            elif qidx == 5:
                repetition = torch.tensor(iceVal, dtype=torch.float32).view(-1, 1)
            elif qidx == 6:
                polarity = torch.tensor(iceVal, dtype=torch.float32).view(-1, 1)
            elif qidx == 7:
                subjectivity = torch.tensor(iceVal, dtype=torch.float32).view(-1, 1)
            elif qidx == 8:
                featureAppearance = torch.tensor(iceVal, dtype=torch.float32).view(-1, 1)
            elif qidx == 9:
                grammar = torch.tensor(iceVal, dtype=torch.float32).view(-1, 1)

            dataset[convIdx] = input_ids,attention_masks,length,readability,wordImp,\
                    repetition,subjectivity,polarity,grammar,featureAppearance,labels
            
            prediction = model.predict(dataset[convIdx])
            convPrediction.append(prediction)
        totalPredictions.append(convPrediction)
    return totalPredictions

def main():
    testidList,testSeekerConv,testRecommenderConv,testLength,testReadability,testWordImp,testRepetition,testSubjectivity,testPolarity,\
        testGrammar,testFeatureAppearance,testpreservedOrder = readInData('test')

    #Get label data for train / test
    testLabels = readInLabels('test')   
    #Pad input data so all conversations are the same length
    testInput = padConversation(testpreservedOrder)
    testDataset = encoderNetwork(tokenizer,testInput,testLength,testReadability,testWordImp,testRepetition,testSubjectivity,testPolarity,testGrammar,testFeatureAppearance,testLabels) 

    #encoder classifier model
    classifier = ClassifierNetwork.load_from_checkpoint("",model=model)
    
    icedata = {}
    for i in range(2,10): #2,3,4,5,6,7,8,9, -->indexes of quality scores in dataloader
        if i ==2:
            qname='length'
        elif i ==3:
            qname='readability'    
        elif i ==4:
            qname='wordimportance'
        elif i ==5:
            qname='repetition'        
        elif i ==6:
            qname='polarity'        
        elif i ==7:
            qname='subjectivity'        
        elif i ==8:
            qname='featureappearance'
        elif i ==9:
            qname='grammar'
        icedata[qname] = iceAnalysis(i,testDataset,classifier)

    with open('gpt2icedata.pkl','wb') as f:
        pickle.dump(icedata,f)

if __name__ == "__main__":
    main()

