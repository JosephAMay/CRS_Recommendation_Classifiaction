!/usr/bin/env python3
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
from readInDataAndClassifyDData.py import readInData,readInLabels

# Check system requirements
if torch.cuda.is_available():
    print(torch.cuda.get_device_name())
    print(torch.cuda.get_device_properties("cuda"))
    print("Number of devices:", torch.cuda.device_count())
    device = ("cuda")
else:
    print("Only CPU is available...")
    device = ("cpu")


class EncoderClassifier(pl.LightningModule):
    def __init__(self, tokenizer, gpt2_model, classifier_input_size):
        super(EncoderClassifier, self).__init__()
        self.tokenizer = tokenizer
        self.gpt2_model = gpt2_model
        self.classifier_input_size = classifier_input_size

        # Classifier network
        self.classifier = nn.Sequential(
            nn.Linear(self.gpt2_model.config.hidden_size + self.classifier_input_size, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 3),  # Output: Bad, Okay, Good
            nn.Softmax(dim=1)   # Softmax for multi-class classification
        )

    def forward(self, conversation, additional_features):
        # Tokenize and encode conversation using GPT-2
        encoded_conversation = self.gpt2_model(conversation)['last_hidden_state'][:, 0, :]
        
        # Concatenate GPT-2 encoded conversation with additional features
        concatenated_input = torch.cat([encoded_conversation, additional_features], dim=1)
        
        # Forward pass through the classifier network
        output = self.classifier(concatenated_input)
        return output

    def training_step(self, batch, batch_idx):
        conversation, additional_features, labels = batch
        output = self(conversation, additional_features)
        loss = nn.CrossEntropyLoss()(output, labels)
        return loss

    def validation_step(self, batch, batch_idx):
        conversation, additional_features, labels = batch
        output = self(conversation, additional_features)
        loss = nn.CrossEntropyLoss()(output, labels)
        pred_labels = torch.argmax(output, dim=1)
        correct = (pred_labels == labels).sum().item()
        return {'val_loss': loss, 'correct': correct, 'total': labels.size(0)}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        total_correct = sum(x['correct'] for x in outputs)
        total_samples = sum(x['total'] for x in outputs)
        val_acc = total_correct / total_samples
        self.log('val_loss', avg_loss)
        self.log('val_acc', val_acc)

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.001)


# Custom dataset class
class CustomDataset(Dataset):
    def __init__(self, preserved_order, additional_features, labels):
        self.preserved_order = preserved_order
        self.additional_features = additional_features
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        conversation = self.preserved_order[idx]
        additional_feature = self.additional_features[idx]
        label = self.labels[idx]
        return conversation, additional_feature, label

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
    gpt2_model = GPT2Model.from_pretrained('gpt2')

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

    train_dataset = CustomDataset(train_preserved_order, train_additional_features, train_labels)
    test_dataset = CustomDataset(test_preserved_order, test_additional_features, test_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize the encoder classifier model
    encoder_classifier = EncoderClassifier(tokenizer_gpt2, gpt2_model, classifier_input_size=train_additional_features.size(1))

    # Initialize the CSV Logger
    logger = pl.loggers.CSVLogger("lightning_logs", name="ClassifierTest", version="gpt2")

    # Initialize the trainer
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=5,
        gpus=1,  # You can set the number of GPUs
        enable_progress_bar=True,
        log_every_n_steps=0,
        enable_checkpointing=True,
        callbacks=[pl.callbacks.TQDMProgressBar(refresh_rate=50)]
    )

    # Train the model
    trainer.fit(encoder_classifier, train_dataloader, test_dataloader)

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
