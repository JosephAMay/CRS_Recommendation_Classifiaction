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


# Test data 756 long. 756 / 18 == 42 even runs through data.
BATCH_SIZE = 4
NUM_CLASSES=3
NUM_EPOCHS = 60

# Load pre-trained GPT-2 model and tokenizer
gpt2Model = GPT2Model.from_pretrained('gpt2')
gpt2Tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2Tokenizer.add_special_tokens({'pad_token': '[PAD]'})


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

    max_length = max(len(ids[0]) for ids in input_ids)
    
    
    input_ids = pad_sequence([F.pad(ids, pad=(0, max_length - ids.size(1)), value=0) for ids in input_ids], batch_first=True)
    attention_masks = pad_sequence([F.pad(masks, pad=(0, max_length - masks.size(1)), value=0) for masks in attention_masks], batch_first=True)

    
    length = torch.tensor(length, dtype=torch.float32).view(-1, 1)
    readability = torch.tensor(readability, dtype=torch.float32).view(-1, 1)
    wordImp = torch.tensor(wordImp, dtype=torch.float32).view(-1, 1)
    repetition = torch.tensor(repetition, dtype=torch.float32).view(-1, 1)
    subjectivity = torch.tensor(subjectivity, dtype=torch.float32).view(-1, 1)
    polarity = torch.tensor(polarity, dtype=torch.float32).view(-1, 1)
    grammar = torch.tensor(grammar, dtype=torch.float32).view(-1, 1)
    featureAppearance = torch.tensor(featureAppearance, dtype=torch.float32).view(-1, 1)
    #labels = torch.nn.functional.one_hot(torch.tensor(labels), num_classes=NUM_CLASSES)
    labels = torch.tensor(labels,dtype=torch.long)

    print('label is', labels)

    
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

        label = torch.tensor(int(self.labels[idx]))

        return input_ids, attention_mask, lenScore,readScore,wordScore,repScore,polScore,subScore,featScore,gramScore, label

# Classify the conversation
class ClassifierNetwork(pl.LightningModule):
    def __init__(self, gpt2_model, num_classes=NUM_CLASSES):
        super(ClassifierNetwork, self).__init__()

        # Freeze GPT-2 weights
        for param in gpt2_model.parameters():
            param.requires_grad = False

        self.gpt2 = gpt2_model
        self.fc1 = nn.Linear(776, 256)
        self.fc2 = nn.Linear(256, 128)      
        self.fc3 = nn.Linear(128, NUM_CLASSES)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.classification.Accuracy(num_classes=NUM_CLASSES, task='multiclass')


    def forward(self, input_ids, attention_mask, lenScore,readScore,wordScore,repScore,polScore,subScore,featScore,gramScore):
        #input_ids = input_ids[0]
        #attention_mask = attention_mask[0]
        outputs = self.gpt2(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs['last_hidden_state']  # Use last hidden states instead of pooler_output
        #print("Shapes:", lenScore.shape, readScore.shape, wordScore.shape, repScore.shape, polScore.shape, subScore.shape, featScore.shape, gramScore.shape)

        # Take the mean of last hidden states along the sequence dimension
        pooled_output = torch.mean(last_hidden_states, dim=1)
        
        
        scores = torch.cat((lenScore, readScore, wordScore, repScore, polScore, subScore, featScore, gramScore), dim=1)
        #scores = scores.view(1, -1)  # Reshape to have 1 row and as many columns as needed to combine with encoded CONV
        
        #print("Pooled Output Shape:", pooled_output.shape)
        #print("Scores Shape:", scores.shape)
        #print(scores)
        #print("Pooled Output Shape:", pooled_output.shape)
        
        
        
        combined_outputs = []

        # Iterate over each instance in the batch
        for i in range(pooled_output.size(0)):
            # Select the pooled output and scores for the current instance
            current_pooled_output = pooled_output[i, :, :]
            current_scores = scores[i, :]
        
            # Repeat the scores to match the sequence length
            repeated_scores = current_scores.repeat(current_pooled_output.size(0), 1)
        
            # Concatenate pooled output with repeated scores along dimension 1
            combined_output = torch.cat((current_pooled_output, repeated_scores), dim=1)
        
            # Append to the list
            combined_outputs.append(combined_output)
        
        # Concatenate along dimension 0 to maintain batch integrity
        combined_outputs = torch.stack(combined_outputs, dim=0)

        #print("Combined Output Shape:", combined_outputs.shape)
        #print("Scores Shape:", scores.shape)
        
        
        # Apply linear layers
        x = F.relu(self.fc1(combined_outputs))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)

        logits = torch.mean(logits, dim=1)

        return logits

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, lenScore, readScore, wordScore, repScore, polScore, subScore, featScore, gramScore, labels = batch
        logits = self(input_ids, attention_mask, lenScore, readScore, wordScore, repScore, polScore, subScore, featScore, gramScore)
        print('Logits in training step:',logits.shape)
        loss = self.loss_fn(logits, labels)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        print('Line 260, predictions are:', torch.unique(preds))
        acc = self.accuracy(preds, labels)

        # Log metrics
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, lenScore, readScore, wordScore, repScore, polScore, subScore, featScore, gramScore, labels = batch
        logits = self(input_ids, attention_mask, lenScore, readScore, wordScore, repScore, polScore, subScore, featScore, gramScore)
        print('Logits in VALIDATIOn step:',logits.shape)
        loss = self.loss_fn(logits, labels)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        print('Line 275, predictions are:', torch.unique(preds))
        acc = self.accuracy(preds, labels)

        # Log metrics
        self.log('val_acc', acc, on_step=False, on_epoch=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        return torch.optim.AdamW(self.parameters(), lr=2e-5)



def main():
    #Input files used 
    #trainfilename = 'TRAIN_combinedData.txt'
    #testfilename = 'TEST_combinedData.txt'

    #Target label files used
    #trainTargetFilename = 'TRAIN_Labels_CombinedData.csv'
    #testTargetFilename = 'TEST_Labels_CombinedData.csv'

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

    
    
    
    trainDataset = encoderNetwork(gpt2Tokenizer,trainInput, trainLength,trainReadability,trainWordImp,trainRepetition,trainSubjectivity,trainPolarity,trainGrammar,trainFeatureAppearance, trainLabels) 
    testDataset = encoderNetwork(gpt2Tokenizer,testInput,testLength,testReadability,testWordImp,testRepetition,testSubjectivity,testPolarity,testGrammar,testFeatureAppearance,testLabels) 

    trainDataloader = DataLoader(trainDataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=customCollate)
    testDataloader = DataLoader(testDataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=customCollate)

    #Initialize the CSV Logger for stat tracking
    logger = pl.loggers.CSVLogger("lightning_logs", name="ClassifierTest", version="gpt2")

    #encoder classifier model
    classifier = ClassifierNetwork(gpt2Model)

    
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

