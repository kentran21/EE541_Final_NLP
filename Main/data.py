import csv
import sys
import torch
import pickle
import matplotlib.pyplot as plt
from os import listdir
from os.path import isfile, join
from transformers import BertTokenizer, BertModel
from tqdm import tqdm
from collections import defaultdict

FILE_DIR_TRAIN = "../Data/train.csv"
FILE_DIR_TEST = "../Data/test.csv"

# Bert tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
# Load pre-trained model
model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states = False)
# maximum length of tokens
T = 50

# only for demonstration
def token_lengths(file_name: str):
    data = defaultdict(dict)
    with open(file_name, "r", encoding="utf8") as f:
        reader = csv.reader(f, delimiter=",")
        ids_save = []
        token_lengths = []
        count = 0
        count_lessthan = 0
        for i, line in enumerate(reader):
            if i > 0:
                marked_utterance = "[CLS] " + line[3]
                tokenized_utterance = tokenizer.tokenize(marked_utterance)
                token_lengths.append(len(tokenized_utterance))
                count = count + 1
                if len(tokenized_utterance) < 50:
                    count_lessthan = count_lessthan + 1
        print(count_lessthan/count)
        plt.hist(token_lengths, density=False, bins=30) 
        plt.ylabel('Probability')
        plt.xlabel('Data')
        plt.show()


def process_sentences(file_name: str, train = True):
    data = defaultdict(dict)
    with open(file_name, "r", encoding="utf8") as f:
        reader = csv.reader(f, delimiter=",")
        ids_save = []
        for i, line in enumerate(reader):
            if i > 0:
                data[line[0]]['keyword'] = line[1]
                data[line[0]]['location'] = line[2]
                data[line[0]]['text'] = line[3]
                if train:
                    data[line[0]]['target'] = line[4]
                data[line[0]]['embedding'] = Bert_embed(line[3])
                ids_save.append(line[0])
            if i % 20 == 0:
                print(i, "lines done")
    return data, ids_save

def Bert_embed(sentence: str):
    # Tokenize the sentence
    marked_sentence = "[CLS] " + sentence
    tokenized_sentence = tokenizer.tokenize(marked_sentence)
    # truncate or pad
    if len(tokenized_sentence) >= T:
        fixed_tokens = tokenized_sentence[:T]
        attention_mask = [1] * T

    else:
        fixed_tokens = ["[PAD]" for i in range(T)]
        fixed_tokens[:len(tokenized_sentence)] = tokenized_sentence
        attention_mask = [1] * len(tokenized_sentence) + [0] * (T - len(tokenized_sentence))
    indexed_tokens = tokenizer.convert_tokens_to_ids(fixed_tokens)
    # Convert inputs to PyTorch tensors
    tokens_tensor = torch.tensor([indexed_tokens])
    attention_tensor = torch.tensor([attention_mask])
    # get embedding
    model.eval()
    with torch.no_grad():
        outputs = model(tokens_tensor, attention_mask = attention_tensor)
    # output (all tokens of last layer) is of shape 768 x 50
    output = torch.transpose(outputs[0][0, :, :], 0, 1)
    return output

if __name__ == '__main__':
    # token_lengths(FILE_DIR_TRAIN)

    val = input("train data or test data, enter 'train' or 'test': ")
    if val == 'train':
        data, ids = process_sentences(FILE_DIR_TRAIN)
        # save
        with open("../Data/dict_train.pkl", "wb") as tf:
            pickle.dump(data,tf)
        with open("../Data/ids_train.pkl", "wb") as tf:
            pickle.dump(ids,tf)
    elif val == 'test':
        data, ids = process_sentences(FILE_DIR_TEST, False)
        # save
        with open("../Data/dict_test.pkl", "wb") as tf:
            pickle.dump(data,tf)
        with open("../Data/ids_test.pkl", "wb") as tf:
            pickle.dump(ids,tf)
    else:
        print("invalid input")
    
    
    

        