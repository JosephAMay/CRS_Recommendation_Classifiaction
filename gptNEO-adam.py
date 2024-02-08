#!/usr/bin/env python3
import numpy as np
import torch
import pytorch_lightning as pl
import torchmetrics
import torchvision
import torch.nn as nn
import torch.optim as optim
from transformers import GPTNeoModel, GPT2Tokenizer
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
import sys
import random


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
        filename= 'TRAIN_combinedData.txt'
    else:
        filename= 'TEST_combinedData.txt'
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
    
# Pad sequences within each batch got some IO errors on data size so this fixes that
def customCollate(batch):
    input_ids, attention_masks,length,readability,wordImp,repetition,subjectivity,\
        polarity,grammar,featureAppearance, labels = zip(*batch)

    #Find biggerst conversation in the batch
    max_length = max(len(ids[0]) for ids in input_ids)
    
    #Pad each conversation inputids and attentionMasks to max_length so they all have the same dimensions
    input_ids = pad_sequence([F.pad(ids, pad=(0, max_length - ids.size(1)), value=0) for ids in input_ids], batch_first=True)
    attention_masks = pad_sequence([F.pad(masks, pad=(0, max_length - masks.size(1)), value=0) for masks in attention_masks], batch_first=True)

    #Paddings adds a singleton dimension that causes errors down the line. Squeeze dimension out so that IDS and masks are 
    #[batchSize,sequenceLength] rather than [batchSize,1,sequenceLength]
    input_ids = input_ids.squeeze(dim=1)
    attention_masks = attention_masks.squeeze(dim=1)

    #reshape each quality factor so it is a tensor with 1 dimension    
    length = torch.tensor(length, dtype=torch.float32).view(-1, 1)
    readability = torch.tensor(readability, dtype=torch.float32).view(-1, 1)
    wordImp = torch.tensor(wordImp, dtype=torch.float32).view(-1, 1)
    repetition = torch.tensor(repetition, dtype=torch.float32).view(-1, 1)
    subjectivity = torch.tensor(subjectivity, dtype=torch.float32).view(-1, 1)
    polarity = torch.tensor(polarity, dtype=torch.float32).view(-1, 1)
    grammar = torch.tensor(grammar, dtype=torch.float32).view(-1, 1)
    featureAppearance = torch.tensor(featureAppearance, dtype=torch.float32).view(-1, 1)
    labels = torch.tensor(labels,dtype=torch.long)

    return input_ids,attention_masks,length,readability,wordImp,\
            repetition,subjectivity,polarity,grammar,featureAppearance, labels
  
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
        
        lenScore = torch.tensor(self.length[idx])
        readScore = torch.tensor(self.readability[idx])
        wordScore = torch.tensor(self.wordImp[idx])
        repScore = torch.tensor(self.repetition[idx])
        polScore =  torch.tensor(self.polarity[idx])
        subScore =  torch.tensor(self.subjectivity[idx])
        featScore =  torch.tensor(self.featApp[idx])
        gramScore= torch.tensor(self.grammar[idx])

        label = torch.tensor(int(self.labels[idx]),dtype=torch.long)

        return input_ids, attention_mask, lenScore,readScore,wordScore,repScore,polScore,subScore,featScore,gramScore, label

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



def main():
    #Keep track of each independent run to load models for later
    if len(sys.argv) >1:
        versionNum = sys.argv[1]
    else: #If I forgot to send in a version, try and generate a unique Vnum.
        versionNum = random.randint(300,600)
    #Make sure to specify input files used in readin data.
    #Get input data from file, this includes the conversation, and its associated quality factors
    trainidList,trainseekerConv,trainrecommenderConv,trainLength,trainReadability,trainWordImp,trainRepetition,trainSubjectivity,trainPolarity, \
        trainGrammar,trainFeatureAppearance,trainpreservedOrder = readInData('training')
    testidList,testSeekerConv,testRecommenderConv,testLength,testReadability,testWordImp,testRepetition,testSubjectivity,testPolarity,\
        testGrammar,testFeatureAppearance,testpreservedOrder = readInData('test')

    #Get label data for train / test
    trainLabels = readInLabels('training')
    testLabels = readInLabels('test')

   
    #Pad input data so all conversations are the same length
    trainInput = padConversation(trainpreservedOrder)
    testInput = padConversation(testpreservedOrder)

    
    
    #Get data, encode it, pass to dataloader
    trainDataset = encoderNetwork(tokenizer,trainInput, trainLength,trainReadability,trainWordImp,trainRepetition,trainSubjectivity,trainPolarity,trainGrammar,trainFeatureAppearance, trainLabels) 
    testDataset = encoderNetwork(tokenizer,testInput,testLength,testReadability,testWordImp,testRepetition,testSubjectivity,testPolarity,testGrammar,testFeatureAppearance,testLabels) 

    trainDataloader = DataLoader(trainDataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=customCollate,num_workers=8)
    testDataloader = DataLoader(testDataset, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=customCollate,num_workers=8)

    #Initialize the CSV Logger for stat tracking
    logger = pl.loggers.CSVLogger("lightning_logs", name="ClassifierTest", version="NEO-v"+str(versionNum))

    #encoder classifier model
    classifier = ClassifierNetwork(model)

    
    #Initialize the trainer
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=NUM_EPOCHS,
        enable_progress_bar=True,
        log_every_n_steps=0,
        enable_checkpointing=True,
        callbacks=[pl.callbacks.TQDMProgressBar(refresh_rate=50)]
    )

    # Train the model
    trainer.fit(classifier, trainDataloader, testDataloader)

    #Read in metrics from metrics file. Output metrics to stdout
    results = pd.read_csv(logger.log_dir+"/metrics.csv")

    print("Validation accuracy:",*["%.8f"%(x) for x in
         results['val_acc'][np.logical_not(np.isnan(results["val_acc"]))]])

    
    print("Validation loss:",*["%.8f"%(x) for x in
         results['val_loss'][np.logical_not(np.isnan(results["val_loss"]))]])

    print("Test Accuracy:",*["%.8f"%(x) for x in
         results['train_acc'][np.logical_not(np.isnan(results["train_acc"]))]])

    print("Training loss:",*["%.8f"%(x) for x in
         results['train_loss'][np.logical_not(np.isnan(results["train_loss"]))]])
    
    

if __name__ == "__main__":
    main()

