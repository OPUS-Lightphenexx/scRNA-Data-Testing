import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset


class Transformer(nn.Module):
    def __init__(self, input_vocab_size, max_length, hidden_size, num_heads, num_layers):
        super(Transformer, self).__init__()
        self.embedding = nn.Embedding(input_vocab_size, hidden_size)
        self.transformer = nn.Transformer(d_model=hidden_size, nhead=num_heads, num_encoder_layers=num_layers,
                                          num_decoder_layers=num_layers)
        self.fc = nn.Linear(hidden_size, input_vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        x = x.permute(1, 0, 2)
        output = self.transformer(x, x)
        output = output.permute(1, 0, 2)
        output = self.fc(output)
        return output


train_sequences = [['A', 'U', 'G', 'C'], ['U', 'G', 'C', 'G'],['A','A','C','A']]
test_sequences = [['A', 'U', 'G', 'C'], ['U', 'G', 'C', 'G'],['A','A','A','A']]
train_labels = [['U', 'G', 'C', 'G'], ['G', 'C', 'A', 'A'],['A','A','C','A']]
token_to_index = {'A': 0, 'U': 1, 'G': 2, 'C': 3}
train_sequences_ind = [[token_to_index[t] for t in seq] for seq in train_sequences]
train_sequences_tensor = [torch.tensor(seq) for seq in train_sequences_ind]
test_seq_ind = [[token_to_index[i] for i in seq] for seq in test_sequences]
test_sequences_tensor = [torch.tensor(seq) for seq in test_seq_ind]
train_labels_ind = [[token_to_index[f] for f in label] for label in train_labels]
train_labels_tensor = [torch.tensor(label) for label in train_labels_ind]
max_length = 10


class SequenceDataset(Dataset):
    def __init__(self, sequences, labels):
        self.sequences = sequences
        self.labels = labels

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        label = self.labels[idx]
        return sequence, label


train_dataset = SequenceDataset(train_sequences_tensor, train_labels_tensor)
test_dataset = SequenceDataset(test_sequences_tensor, [torch.zeros_like(seq) for seq in test_sequences_tensor])
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
hidden_size = 128
num_heads = 4
num_layers = 2
input_vocab_size = len(token_to_index) + 1
model = Transformer(input_vocab_size, max_length, hidden_size, num_heads, num_layers)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())
epochs = 10
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
for epoch in range(epochs):
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        outputs = outputs.view(-1, input_vocab_size)
        targets = targets.view(-1)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
model.eval()
predictions = []
with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 2)
        predictions.extend(predicted.squeeze().tolist())
predicted_tokens = [[list(token_to_index.keys())[list(token_to_index.values()).index(token)] for token in seq] for seq in predictions]
print(predicted_tokens)

sys.stdout.flush()