#!/usr/bin/env python3
import numpy as np
import torch
import pytorch_lightning as pl
import torchmetrics
import torchvision
from torchinfo import summary
from torchview import draw_graph
from IPython.display import display
import torch.nn as nn
import torch.optim as optim
from transformers import GPT2Tokenizer, GPT2Model
from torch.utils.data import DataLoader, Dataset
import json
from torch.nn.utils.rnn import pad_sequence
import torch.nn.functional as F
import pandas as pd

# Test data 756 long. 756 / 18 == 42 even runs through data.
TRAIN_BATCH_SIZE = 4
TEST_BATCH_SIZE = 5
NUM_EPOCHS = 60

# Load pre-trained GPT-2 model and tokenizer
model_gpt2 = GPT2Model.from_pretrained('gpt2')
tokenizer_gpt2 = GPT2Tokenizer.from_pretrained('gpt2')
tokenizer_gpt2.add_special_tokens({'pad_token': '[PAD]'})

# Check system requirements
if torch.cuda.is_available():
    print(torch.cuda.get_device_name())
    print(torch.cuda.get_device_properties("cuda"))
    print("Number of devices:", torch.cuda.device_count())
    device = ("cuda")
else:
    print("Only CPU is available...")
    device = ("cpu")


# Get the input data from the file
def get_input_data(filename):
    # Open jsonFile
    with open(filename, 'r', encoding='utf-8') as f:
        messages = json.load(f)

    # Loop through message and grab sentences
    forPretrainConv = []
    convRoleList = []
    for message in messages:
        preservedOrderList = []
        cur_roles = []
        workin = message['messages']
        for dictionary in workin:
            role = dictionary['role']

            # Adjust 0 role to 2 for NN processing later
            if role == 0:
                role = 2
            cur_roles.append(role)
            sentence = dictionary['text']
            preservedOrderList.append(sentence)  # Add sentence to this conversation

        # add conversation to list of conversations
        convRoleList.append(cur_roles)
        forPretrainConv.append(preservedOrderList)
        joinedConversations = combineConsecutiveSpeakerSentences(forPretrainConv, convRoleList)
    # return forPretrainConv, convRoleList
    return joinedConversations


def combineConsecutiveSpeakerSentences(input_lists, roles_list):
    joined_strings_list = []

    for input_list, role_list in zip(input_lists, roles_list):
        joined_strings = []
        current_role = None
        current_string = ""

        for text, role in zip(input_list, role_list):
            if current_role is None:
                # First iteration
                current_role = role
                current_string = text
            elif current_role == role:
                # Same role, concatenate the strings
                current_string += " " + text
            else:
                # Different role, start a new string
                joined_strings.append(current_string)
                current_role = role
                current_string = text

        # Append the last string
        joined_strings.append(current_string)
        joined_strings_list.append(joined_strings)

    return joined_strings_list

# Get target data from the file  # Data stored as convID,target
def get_target_data(filename):
    targets = []
    with open(filename, 'r') as file:
        for line in file:
            values = line.strip().split(',')
            # Ignore convID, just grab target label, convert to tensor
            targets.append(torch.tensor(int(values[1])))

    return np.array(targets)


# Function to pad conversations to a common length
def pad_conversations(conversations):
    # Get maxlength conversation, and pad each conversation so they're the same length
    max_length = max(len(conv) for conv in conversations)
    padded_conversations = []
    for conv in conversations:
        padded_conv = conv + [''] * (max_length - len(conv))
        padded_conversations.append(padded_conv)
    return padded_conversations


class encoderNetwork(Dataset):
    def __init__(self, conversations, targets, tokenizer):
        self.conversations = conversations  # data
        # self.roles = roles  # Role information
        self.targets = targets  # Target Labels
        self.tokenizer = tokenizer  # GPT-2 tokenizer

    def __len__(self):
        return len(self.conversations)

    # Convert from list of sentence strings to one long string, encode the string
    def __getitem__(self, idx):
        
        text = " ".join(self.conversations[idx])
        encoding = self.tokenizer(text, padding=True, truncation=True, return_tensors="pt")

        input_ids = encoding["input_ids"].squeeze()
        attention_mask = encoding["attention_mask"].squeeze()
        label = torch.tensor(self.targets[idx])

        return input_ids, attention_mask, label

# Classify the conversation
class ClassifierNetwork(pl.LightningModule):
    def __init__(self, gpt2_model, max_conv_length):
        super(ClassifierNetwork, self).__init__()

        # Freeze GPT-2 weights
        for param in gpt2_model.parameters():
            param.requires_grad = False

        self.gpt2 = gpt2_model
        self.fc1 = nn.Linear(self.gpt2.config.hidden_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 2)
        self.loss_fn = nn.CrossEntropyLoss()
        self.accuracy = torchmetrics.classification.Accuracy(task='binary')

    def forward(self, input_ids, attention_mask):
        outputs = self.gpt2(input_ids, attention_mask=attention_mask)
        last_hidden_states = outputs['last_hidden_state']  # Use last hidden states instead of pooler_output

        # Take the mean of last hidden states along the sequence dimension
        pooled_output = torch.mean(last_hidden_states, dim=1)

        # Apply linear layers
        x = F.relu(self.fc1(pooled_output))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)

        return logits

    def training_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
        loss = self.loss_fn(logits, labels)

        # Calculate accuracy
        preds = torch.argmax(logits, dim=1)
        acc = self.accuracy(preds, labels)

        # Log metrics
        self.log('train_acc', acc, on_step=False, on_epoch=True)
        self.log('train_loss', loss, on_step=False, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        input_ids, attention_mask, labels = batch
        logits = self(input_ids, attention_mask)
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

# Pad sequences within each batch got some IO errors on data size so this fixes that
def custom_collate_fn(batch):
    input_ids, attention_masks, labels = zip(*batch)

    # Determine the maximum length in the batch
    max_len = max(len(ids) for ids in input_ids)

    # Pad or truncate sequences to the maximum length using the padding token (0 in this case)
    padded_input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    padded_attention_masks = pad_sequence(attention_masks, batch_first=True, padding_value=0)

    return padded_input_ids, padded_attention_masks, torch.tensor(labels)



def main():
    # Setup input files.
    trainfilename = 'eredial.json'
    testfilename = 'test.json'

    #Setup target files
    trainTargetFilename = 'E-redial-TRAINING-LABELS.txt'
    testTargetFilename = 'E-redial-TEST-LABELS.txt'

    #GEt input data from file
    trainInput = get_input_data(trainfilename)  #, trainRoles = get_input_data(trainfilename)
    testInput = get_input_data(testfilename)  #, testRoles = get_input_data(testfilename)

    #get maximum conversation Lengths
    max_conv_length = max(len(conv) for conv in trainInput + testInput)
    

    #Pad conversations and roles so they have the same length
    trainInput = pad_conversations(trainInput)
    testInput = pad_conversations(testInput)

    #Get label data for train / test
    trainLabels = get_target_data(trainTargetFilename)
    testLabels = get_target_data(testTargetFilename)

    #Embed the data
    train_dataset = encoderNetwork(trainInput, trainLabels, tokenizer_gpt2)
    test_dataset = encoderNetwork(testInput,  testLabels, tokenizer_gpt2)

    #Make the dataloaders loggers and trainers
    train_dataloader = DataLoader(train_dataset, batch_size=TRAIN_BATCH_SIZE, shuffle=True, collate_fn=custom_collate_fn, num_workers=2)
    test_dataloader = DataLoader(test_dataset, batch_size=TEST_BATCH_SIZE, shuffle=False, collate_fn=custom_collate_fn, num_workers=2)

    logger = pl.loggers.CSVLogger("lightning_logs", name="ClassifierTest", version="gpt2")

    #Make classifier network
    classifier_network = ClassifierNetwork(model_gpt2, max_conv_length)

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=NUM_EPOCHS,
        enable_progress_bar=True,
        log_every_n_steps=0,
        enable_checkpointing=True,
        callbacks=[pl.callbacks.TQDMProgressBar(refresh_rate=50)]
    )

    #Train up and run code
    trainer.fit(classifier_network, train_dataloader, test_dataloader)

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
