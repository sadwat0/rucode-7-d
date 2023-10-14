import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm, trange
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from typing import List
from IPython.display import clear_output

LETTERS = "qабвгдежзийклмнопрстуфхцчшщъыьэюяё"
VOWELS = "аеиоуэюяыё"
N_LETTERS = len(LETTERS)
MAX_WORD_LENGTH = 36

char2ind = {char: i for i, char in enumerate(LETTERS)}
ind2char = {i: char for char, i in char2ind.items()}

# test_private = open('drive/MyDrive/rucode-D data/private_test_stresses.txt', 'r', encoding="utf8").readlines()
test_private = open('data/private_test_stresses.txt', 'r', encoding="utf8").readlines()
for i in range(len(test_private)):
    test_private[i] = test_private[i][:-1]
    
class CustomDataset:
    def __init__(self, X, y):
        self.data = []
        self.pad_id = char2ind['q']

        assert len(X) == len(y), "X and y must be same length"
        for word, label in zip(X, y):
            self.data.append({
                'word': word,
                'label': label
            })

    def __getitem__(self, idx: int) -> List[int]:
        preprocessed_word = [char2ind[char] for char in self.data[idx]['word']]

        train_sample = {
            "text": preprocessed_word,
            "label": self.data[idx]['label']
        }

        return train_sample

    def __len__(self) -> int:
        return len(self.data)
    
device = 'cuda' if torch.cuda.is_available() else 'cpu'
def collate_fn_with_padding(input_batch: List[List[int]], pad_id=char2ind['q'], max_len=MAX_WORD_LENGTH) -> torch.Tensor:
    words_lengths = [len(x['text']) for x in input_batch]
    # max_word_length = min(max(words_lengths), max_len)
    max_word_length = max_len

    new_batch = []
    for sequence in input_batch:
        sequence['text'] = sequence['text'][:max_word_length]
        for _ in range(max_word_length - len(sequence['text'])):
            sequence['text'].append(pad_id)

        new_batch.append(sequence['text'])

    words = torch.LongTensor(new_batch).to(device)
    labels = torch.LongTensor([x['label'] for x in input_batch]).to(device)

    new_batch = {
        'input_ids': words,
        'label': labels
    }

    return new_batch

class RNN(nn.Module):
    def __init__(self, hidden_dim: int, aggregation_type: str = 'weighted'):
        super().__init__()
        self.embedding = nn.Embedding(N_LETTERS, hidden_dim)
        self.rnn = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, num_layers=2, dropout=0.1, bidirectional=True)
        self.linear = nn.Linear(2 * hidden_dim, hidden_dim)
        self.projection = nn.Linear(2 * hidden_dim, N_LETTERS)

        self.non_lin = nn.Tanh()
        self.dropout = nn.Dropout(p=0.15)

        self.aggregation_layer = torch.nn.Conv1d(in_channels=MAX_WORD_LENGTH, out_channels=1, kernel_size=1)
        self.aggregation_layer2 = torch.nn.Conv1d(in_channels=MAX_WORD_LENGTH, out_channels=1, kernel_size=1)
        self.aggregation_type = aggregation_type

    def forward(self, input_batch) -> torch.Tensor:
        embeddings = self.embedding(input_batch)
        output, _ = self.rnn(embeddings)

        batch_size = output.shape[0]

        if self.aggregation_type == 'max':
            output = output.max(dim=1)[0]
        elif self.aggregation_type == 'mean':
            output = output.mean(dim=1)
        elif self.aggregation_type == 'weighted':
            output = self.aggregation_layer(output).view(batch_size, -1)
        else:
            raise ValueError("Invalid aggregation_type")

        embeddings = self.aggregation_layer2(embeddings).view(batch_size, -1)
        output = self.dropout(self.linear(self.non_lin(output)))
        # print(embeddings.shape, output.shape)

        output = torch.cat((output, embeddings), dim=1)
        # print(output.shape)
        # assert 0
        prediction = self.projection(self.non_lin(output))

        return prediction

model = torch.load('model_8').to(device)

X_test_private = test_private
y_test_private = range(len(X_test_private))

model.eval()

test_private_files = CustomDataset(X_test_private, y_test_private)
test_private_loader = DataLoader(test_private_files, collate_fn=collate_fn_with_padding, batch_size=128, shuffle=False)

res_private = [0] * len(y_test_private)

for batch in test_private_loader:
    with torch.no_grad():
        logits = model(batch['input_ids'])
        preds = torch.argmax(logits, 1)

        for prediction, pos in zip(preds, batch['label']):
            res_private[pos] = prediction.item()

with open(f'train_log/predictions_private_8.txt', 'w', encoding="utf8") as f:
    for i, x in enumerate(res_private):
        f.write(f"{test_private[i][:x]}^{test_private[i][x:]}\n")
