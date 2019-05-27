import torch
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import unicodedata
import csv
import numpy
import copy

torch.manual_seed(0)
torch.backends.cudnn.deterministic=True
torch.backends.cudnn.benchmark=False

# Class to pad variable length sequence according to biggest sequence in the batch
class PadSequence:
    def __call__(self, batch):
        sequences_s1 = [x[0] for x in batch]
        sequences_s1_padded = torch.nn.utils.rnn.pad_sequence(sequences_s1, batch_first=True)
        sequences_s2 = [x[1] for x in batch]
        sequences_s2_padded = torch.nn.utils.rnn.pad_sequence(sequences_s2, batch_first=True)        
        lengths_s1 = torch.LongTensor([len(x) for x in sequences_s1])
        lengths_s2 = torch.LongTensor([len(x) for x in sequences_s2])
        labels = torch.stack([x[2] for x in batch])
        return sequences_s1_padded, sequences_s2_padded, labels, lengths_s1, lengths_s2


class StringMatchingDataset(Dataset):
  
    def __init__(self, name, characters, train_split, test_split):
        super(StringMatchingDataset, self).__init__()

        self.data = list(csv.DictReader(open('datasets/{}.csv'.format(name), encoding='utf-8'), delimiter='|', fieldnames=['s1', 's2', 'res']))

        # If we are doing two-fold or not
        if train_split:
            self.data = self.data[:int( len(self.data)  / 2)]
        if test_split:
            self.data = self.data[int( len(self.data)  / 2) :]

        self.characters = characters
        self.n_chars = len(characters)

    def charToIndex(self, char):
        return self.characters.find(char)

    def stringToCharSeq(self, string):
        string = list(bytearray(unicodedata.normalize('NFKD', string), encoding='utf-8'))
        tensor = torch.zeros(len(string), self.n_chars)
        for i, ch in enumerate(string):
            tensor[i][self.characters.index(ch)] = 1.0
        return tensor

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        res = self.data[index]['res']
        if res == 'FALSE' or res == 0 or res == '0':
            res = torch.FloatTensor([1,0])
        else:
            res = torch.FloatTensor([0,1])
        s1 = self.stringToCharSeq(self.data[index]['s1'])
        s2 = self.stringToCharSeq(self.data[index]['s2'])
        return s1, s2, res

def load_dataset(name, characters, two_fold_train, two_fold_test):
    return StringMatchingDataset(name,  characters, two_fold_train, two_fold_test)

def train(model, data, data_val):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    n_epochs = 20
    
    # For early stopping
    patience = 2
    best_loss = float('inf')
    best_results = []
    num_epochs_worse = 0
    
    for epoch in range(n_epochs):
        total_loss = 0.0
        for i, batch in enumerate(data):
            optimizer.zero_grad()
            outputs = model(batch[0].cuda(), batch[1].cuda(), batch[3].cuda(), batch[4].cuda())
            loss = criterion(outputs, batch[2].cuda())
            total_loss += loss.item()
            loss.backward()
            optimizer.step()
        acc, pre, rec, f1 = test(model, data_val)
        print("Epoch {}/{}, Training Loss: {}, Validation Accuracy: {}, Precision: {}, Recall: {}, F1: {}".format(epoch+1,n_epochs, total_loss, acc,pre,rec,f1))
        if total_loss < best_loss:
            best_loss = total_loss
            torch.save(model.state_dict(), 'current_best.pt')
            best_results = [acc, pre, rec, f1]
            num_epochs_worse = 0
        else:
            num_epochs_worse += 1
            if num_epochs_worse == patience:
                print()
                print("------ EARLY STOPPING ------")
                print("Validation accuracy: {}".format(best_results[0]))
                print("Validation precision: {}".format(best_results[1]))
                print("Validation recall: {}".format(best_results[2]))
                print("Validation F1: {}".format(best_results[3]))
                print()
                break

def test(model, data):
    num_true = 0.0
    num_false = 0.0
    num_true_predicted_true = 0.0
    num_true_predicted_false = 0.0
    num_false_predicted_true = 0.0
    num_false_predicted_false = 0.0
    all_outputs = []
    all_labels = []
    with torch.no_grad():
        for batch in data:
            outputs = model(batch[0].cuda(), batch[1].cuda(), batch[3].cuda(), batch[4].cuda())
            for i in range(len(outputs)):
                if batch[2][i][0] == 1:
                    num_false += 1
                    if outputs[i][0] >= outputs[i][1]:
                        num_false_predicted_false += 1
                    else:
                        num_false_predicted_true += 1
                else:
                    num_true += 1
                    if outputs[i][1] >= outputs[i][0]:
                        num_true_predicted_true += 1
                    else:
                        num_true_predicted_false += 1

    acc = (num_true_predicted_true + num_false_predicted_false) / (num_true + num_false)
    try:
        pre = (num_true_predicted_true) / (num_true_predicted_true + num_false_predicted_true)
    except:
        pre = 0
    try:
        rec = (num_true_predicted_true) / (num_true_predicted_true + num_true_predicted_false)
    except:
        rec = 0
    try:
        f1 = 2.0 * ((pre * rec) / (pre + rec))
    except:
        f1 = 0
    return [acc, pre, rec, f1]

def display_results(results1, results2):
    if results2 == None:
        results2 = results1
    print()
    print("------ RESULTS ------")
    print("Accuracy = {}".format((results1[0]+results2[0]) / 2))
    print("Precision = {}".format((results1[1]+results2[1]) / 2))
    print("Recall = {}".format((results1[2]+results2[2]) / 2))
    print("F1 = {}".format((results1[3]+results2[3]) / 2))
    print()
    
def run_model(model, weights, batch_size, data_train, data_test, two_fold):

    data_train = DataLoader(data_train, shuffle=True, num_workers=6, batch_size=int(batch_size), collate_fn=PadSequence())
    data_test = DataLoader(data_test, shuffle=True, num_workers=6, batch_size=int(batch_size), collate_fn=PadSequence())

    # Load weights if the case
    if weights != None:
        print("Loading weights...") # Load, TODO
    else:
        train(model, data_train, data_test)

    results = test(model, data_test)

    if two_fold:
        model.reset_parameters()
        train(model, data_test, data_train)
        results1 = test(model, data_train)
    
    display_results(results, results1 if two_fold else None)
