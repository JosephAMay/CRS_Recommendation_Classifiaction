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
            labels.append(cur[1]) #Label is the 2nd element on the line. Ignore convID

    return labels


class GPT2Encoder(nn.Module):
    def __init__(self, gpt2_model):
        super(GPT2Encoder, self).__init__()
        self.gpt2_model = gpt2_model

    def forward(self, conversation):
        # Tokenize and encode conversation using GPT-2
        encoded_conversation = self.gpt2_model(conversation)['last_hidden_state'][:, -1, :]
        return encoded_conversation


class GPT2Classifier(pl.LightningModule):
    def __init__(self, gpt2_encoder, classifier_input_size):
        super(GPT2Classifier, self).__init__()
        self.gpt2_encoder = gpt2_encoder
        self.classifier_input_size = classifier_input_size

        # Classifier network
        self.classifier = nn.Sequential(
            nn.Linear(self.gpt2_encoder.gpt2_model.config.hidden_size + self.classifier_input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 3),  # Output: Bad, Okay, Good
            nn.Softmax(dim=1)   # Softmax for multi-class classification
        )

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, conversation, additional_features):
        # Encode conversation using GPT2
        encoded_conversation = self.gpt2_encoder(conversation)

        # Concatenate GPT-2 encoded conversation with additional features
        concatenated_input = torch.cat([encoded_conversation, additional_features], dim=1)

        # Forward pass through the classifier network
        output = self.classifier(concatenated_input)
        return output

    def training_step(self, batch, batch_idx):
        conversation, additional_features, labels = batch
        output = self(conversation, additional_features)
        loss = self.loss_fn(output, labels)

        # Calculate accuracy
        preds = torch.argmax(output, dim=1)
        acc = torch.sum(preds == labels).float() / len(labels)

        # Log metrics
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        conversation, additional_features, labels = batch
        output = self(conversation, additional_features)
        loss = self.loss_fn(output, labels)

        # Calculate accuracy
        preds = torch.argmax(output, dim=1)
        acc = torch.sum(preds == labels).float() / len(labels)

        # Log metrics
        self.log('val_acc', acc, on_step=False, on_epoch=True)
        self.log('val_loss', loss, on_step=False, on_epoch=True)

        return {"val_loss": loss, "val_acc": acc}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)

class CustomDataset(Dataset):
    def __init__(self, tokenizer, gpt2_encoder, preserved_order, additional_features, labels):
        self.tokenizer = tokenizer
        self.gpt2_encoder = gpt2_encoder
        self.preserved_order = preserved_order
        self.additional_features = additional_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        conversation = self.preserved_order[idx]

        # Tokenize the conversation using GPT2 tokenizer and convert to tensor
        tokens = self.tokenizer(conversation, return_tensors='pt', truncation=True, padding=True)
        input_ids = tokens['input_ids']

        # Encode conversation using GPT2
        encoded_conversation = self.gpt2_encoder(input_ids)  # Pass input_ids instead of conversation

        additional_feature = self.additional_features[idx]
        label = self.labels[idx]

        return encoded_conversation, additional_feature, label

    @staticmethod
    def collate_fn(batch):
        conversations, additional_features, labels = zip(*batch)

        # Pad sequences to the length of the longest sequence in the batch
        conversations_padded = pad_sequence(conversations, batch_first=True)
        additional_features_tensor = torch.stack(additional_features)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        return conversations_padded, additional_features_tensor, labels_tensor


def main():
    #Input files used 
    #trainfilename = 'TRAIN_combinedData.txt'
    #testfilename = 'TEST_combinedData.txt'

    #Target label files used
    #trainTargetFilename = 'TRAIN_Labels_CombinedData.csv'
    #testTargetFilename = 'TEST_Labels_CombinedData.csv'

    #Get input data from file, this includes the conversation, and its associated quality factors
    trainidList,trainseekerConv,trainrecommenderConv,trainlength,trainreadability,trainwordImp,trainrepetition,trainsubjectivity,trainpolarity,traingrammar,trainfeatureAppearance,trainpreservedOrder = readInData('training')
    testidList,testseekerConv,testrecommenderConv,testlength,testreadability,testwordImp,testrepetition,testsubjectivity,testpolarity,testgrammar,testfeatureAppearance,testpreservedOrder = readInData('test')

    #Get label data for train / test
    trainLabels = readInLabels('training')
    testLabels = readInLabels('test')

    # Example usage
    tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2')
    tokenizer_gpt2.add_special_tokens({'pad_token': '[PAD]'})
    gpt2_model = GPT2Model.from_pretrained('gpt2')
    gpt2_encoder = GPT2Encoder(gpt2_model)

    # Assuming trainLabels and testLabels are torch tensors, and trainpreservedOrder and testpreservedOrder are lists of lists
    train_preserved_order, test_preserved_order = trainpreservedOrder, testpreservedOrder
    train_additional_features = torch.stack([
        torch.tensor(trainlength),
        torch.tensor(trainreadability),
        torch.tensor(trainwordImp),
        torch.tensor(trainrepetition),
        torch.tensor(trainsubjectivity),
        torch.tensor(trainpolarity),
        torch.tensor(traingrammar),
        torch.tensor(trainfeatureAppearance)
    ], dim=1)

    test_additional_features = torch.stack([
        torch.tensor(testlength),
        torch.tensor(testreadability),
        torch.tensor(testwordImp),
        torch.tensor(testrepetition),
        torch.tensor(testsubjectivity),
        torch.tensor(testpolarity),
        torch.tensor(testgrammar),
        torch.tensor(testfeatureAppearance)
    ], dim=1)

    label_encoder = LabelEncoder()
    train_labels = torch.tensor(label_encoder.fit_transform(trainLabels), dtype=torch.long)
    test_labels = torch.tensor(label_encoder.transform(testLabels), dtype=torch.long)

    train_dataset = CustomDataset(tokenizer_gpt2, gpt2_encoder, train_preserved_order, train_additional_features, train_labels)
    test_dataset = CustomDataset(tokenizer_gpt2, gpt2_encoder, test_preserved_order, test_additional_features, test_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=train_dataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=test_dataset.collate_fn)


    # Initialize the encoder classifier model
    gpt2_classifier = GPT2Classifier(gpt2_encoder, classifier_input_size=train_additional_features.size(1))

    # Initialize the CSV Logger
    logger = pl.loggers.CSVLogger("lightning_logs", name="ClassifierTest", version="gpt2")

    # Initialize the trainer
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=5,
        enable_progress_bar=True,
        log_every_n_steps=0,
        enable_checkpointing=True,
        callbacks=[pl.callbacks.TQDMProgressBar(refresh_rate=50)]
    )

    # Train the model
    trainer.fit(gpt2_classifier, train_dataloader, test_dataloader)

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

