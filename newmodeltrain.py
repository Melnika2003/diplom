import pickle
import re
from collections import Counter
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

TRAIN_TEXT_FILE_PATH = '1.txt'

# Read and preprocess text
with open(TRAIN_TEXT_FILE_PATH, encoding='utf-8') as text_file:
    text_sample = text_file.read()  # Read entire file
text_sample = re.sub("[^А-Яа-я0-9\s!-~ёЁ]", "", text_sample)

def text_to_seq(text_sample):
    char_counts = Counter(text_sample)
    char_counts = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_chars = [char for char, _ in char_counts]
    char_to_idx = {char: index for index, char in enumerate(sorted_chars)}
    idx_to_char = {v: k for k, v in char_to_idx.items()}
    sequence = np.array([char_to_idx[char] for char in text_sample])
    return sequence, char_to_idx, idx_to_char

sequence, char_to_idx, idx_to_char = text_to_seq(text_sample)

# Adjusted hyperparameters
SEQ_LEN = 512  # Shorter sequence for ~60-char prompts
BATCH_SIZE = 64  # Larger batch for better GPU utilization
HIDDEN_SIZE = 256  # Smaller model for efficiency
EMBEDDING_SIZE = 256  # Match hidden size
N_LAYERS = 2  # Keep 2 layers for complexity
DROPOUT = 0.3  # Slightly higher to prevent overfitting

def get_batch(sequence):
    trains, targets = [], []
    for _ in range(BATCH_SIZE):
        batch_start = np.random.randint(0, len(sequence) - SEQ_LEN)
        chunk = sequence[batch_start: batch_start + SEQ_LEN]
        train = torch.LongTensor(chunk[:-1]).view(-1, 1)
        target = torch.LongTensor(chunk[1:]).view(-1, 1)
        trains.append(train)
        targets.append(target)
    return torch.stack(trains, dim=0), torch.stack(targets, dim=0)

def evaluate(model, char_to_idx, idx_to_char, start_text='.', prediction_len=60, temp=0.5):
    model.eval()
    hidden = model.init_hidden()
    idx_input = [char_to_idx[char] for char in start_text]
    train = torch.LongTensor(idx_input).view(-1, 1, 1).to(device)
    predicted_text = start_text

    with torch.no_grad():
        _, hidden = model(train, hidden)
        inp = train[-1].view(-1, 1, 1)
        for _ in range(prediction_len):
            output, hidden = model(inp.to(device), hidden)
            output_logits = output.cpu().data.view(-1)
            p_next = F.softmax(output_logits / temp, dim=-1).detach().cpu().numpy()
            top_index = np.random.choice(len(char_to_idx), p=p_next)
            inp = torch.LongTensor([top_index]).view(-1, 1, 1).to(device)
            predicted_text += idx_to_char[top_index]
    return predicted_text

class TextRNN(nn.Module):
    def __init__(self, input_size, hidden_size, embedding_size, n_layers=1):
        super(TextRNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.n_layers = n_layers
        self.encoder = nn.Embedding(self.input_size, self.embedding_size)
        self.lstm = nn.LSTM(self.embedding_size, self.hidden_size, self.n_layers, dropout=DROPOUT if n_layers > 1 else 0)
        self.dropout = nn.Dropout(DROPOUT)
        self.fc = nn.Linear(self.hidden_size, self.input_size)

    def forward(self, x, hidden):
        x = self.encoder(x).squeeze(2)
        out, (ht1, ct1) = self.lstm(x, hidden)
        out = self.dropout(out)
        x = self.fc(out)
        return x, (ht1, ct1)

    def init_hidden(self, batch_size=1):
        return (torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(device),
                torch.zeros(self.n_layers, batch_size, self.hidden_size, requires_grad=True).to(device))

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(torch.cuda.is_available())
model = TextRNN(input_size=len(idx_to_char), hidden_size=HIDDEN_SIZE, embedding_size=EMBEDDING_SIZE, n_layers=N_LAYERS)
model.to(device)

# Optimizer and scheduler
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)  # Lower learning rate
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, verbose=True, factor=0.5)

# Training loop
n_epochs = 120000  # Reduced epochs with early stopping
loss_avg = []
best_loss = float('inf')
patience = 10
patience_counter = 0

# Save character mappings
with open('idx_to_char.pickle', 'wb') as f:
    pickle.dump(idx_to_char, f)
with open('char_to_idx.pickle', 'wb') as f:
    pickle.dump(char_to_idx, f)

for epoch in range(n_epochs):
    model.train()
    train, target = get_batch(sequence)
    train = train.permute(1, 0, 2).to(device)
    target = target.permute(1, 0, 2).to(device)
    hidden = model.init_hidden(BATCH_SIZE)

    output, hidden = model(train, hidden)
    loss = criterion(output.permute(1, 2, 0), target.squeeze(-1).permute(1, 0))

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
    optimizer.step()

    loss_avg.append(loss.item())
    if len(loss_avg) >= 50:  # Check more frequently
        mean_loss = np.mean(loss_avg)
        print(f'Epoch {epoch}, Loss: {mean_loss}')
        scheduler.step(mean_loss)
        loss_avg = []

        # Early stopping
        if mean_loss < best_loss:
            best_loss = mean_loss
            patience_counter = 0
            torch.save(model, 'model.pt')
            torch.save(model.state_dict(), 'modelall.pt')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print("Early stopping triggered")
                break

        # Generate sample output
        model.eval()
        print(evaluate(model, char_to_idx, idx_to_char, start_text='стул ', prediction_len=60, temp=0.5))

model.eval()