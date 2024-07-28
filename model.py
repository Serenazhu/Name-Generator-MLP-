import torch
from torch.nn import functional as F
import tiktoken
import random
# 3 million params(3268849)
# Read the names file
with open('names.txt', 'r') as f:
    t = f.read()
    words = t.split('\n')

# Set device to MPS
device = 'mps'

# Tokenize the words
all_tokens = []
train_tokens = []
val_tokens = []
test_tokens = []
enc = tiktoken.get_encoding('gpt2')
for word in words:
    word = ''.join(['<|start|>'] + list(word) + ['<|end|>'])
    token = enc.encode(word, allowed_special={'<|start|>','<|end|>'})
    all_tokens.append(token)
random.shuffle(all_tokens)
n1 = int(len(all_tokens) * 0.8)
n2 = int(len(all_tokens) * 0.9)
train_tokens = all_tokens[:n1]
val_tokens = all_tokens[n1:n2]
test_tokens = all_tokens[n2:]
# Define vocabulary size
vocab_size = enc.n_vocab

# Create dataset from tokens
def create_ds(tokens):
    xs = []
    ys = []
    for token in tokens:
        for ch1, ch2, ch3 in zip(token, token[1:], token[2:]):
            xs.append(ch1); xs.append(ch2)
            ys.append(ch3)
    xs = torch.tensor(xs, device=device).view(-1, 2)
    ys = torch.tensor(ys, device=device).long()
    return xs, ys

# Define the NLP model
class NLP(torch.nn.Module):
    def __init__(self, vocab_size, embed_size=32):
        super().__init__()
        self.embd = torch.nn.Embedding(vocab_size, embed_size)
        self.layer1 = torch.nn.Linear(embed_size*2, 32)
        self.bn1 = torch.nn.BatchNorm1d(32)
        self.layer2 = torch.nn.Linear(32, vocab_size)
        self.dropout = torch.nn.Dropout(0.8)  # Example dropout rate of 50%

        # Initialize weights properly
        torch.nn.init.xavier_uniform_(self.layer1.weight)
        torch.nn.init.xavier_uniform_(self.layer2.weight)

    def forward(self, x):
        x = self.embd(x)
        x = x.view(x.size(0), -1)  # Flatten the embeddings
        h1 = F.relu(self.bn1(self.layer1(x)))
        h2 = self.layer2(h1)
        return h2

# Create dataset
Xtr, Ytr = create_ds(train_tokens)
Xval, Yval = create_ds(val_tokens)
Xte, Yte = create_ds(test_tokens)
# Initialize model
model = NLP(vocab_size).to(device)

# Training parameters
learning_rate = 0.01  # Reduced learning rate
num_epochs = 40
batch_size = 256  # Increased batch size
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8)
# Training loop
for epoch in range(num_epochs):
    for i in range(0, 5000, batch_size):
        Xbatch = Xtr[i:i+batch_size].to(device)
        Ybatch = Ytr[i:i+batch_size].to(device)

        # Forward Pass
        logits = model(Xbatch)

        loss = F.cross_entropy(logits, Ybatch)

        optimizer.zero_grad()
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Increased gradient clipping threshold
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}')
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0]
    print('Learning Rate: {current_lr:.5f}')
print('finished training!')

def validation():
    model.eval()
    total_loss = 0.0
    num_batces = 0
    with torch.no_grad():
        for i in range(0, len(Xval), batch_size):
            Xbatch = Xval[i:i+batch_size].to(device)
            Ybatch = Yval[i:i+batch_size].to(device)
            # Forward Pass
            logits = model(Xbatch)
            loss = F.cross_entropy(logits, Ybatch)
            total_loss += loss
            num_batces += 1
    averaged_loss = total_loss / num_batces
    return averaged_loss

def sample(model, enc, start_text, max_length=50, device='mps'):
    model.eval()
    input_gen = '<|start|>' + start_text
    start_text_token = enc.encode(start_text) 
    tokens = enc.encode(input_gen, allowed_special={'<|start|>', '<|end|>'}) 
    end_token = enc.encode('<|end|>', allowed_special={'<|start|>', '<|end|>'})[0]
    tokens = torch.tensor(tokens, device=device).view(-1, 1)
    # generated_tokens = start_text_token.squeeze().tolist()
    for _ in range(max_length):
        input_tokens = tokens[-2:].unsqueeze(0)
        logits = model(input_tokens)
        probabilities = F.softmax(logits, dim=1)
        next_token = torch.multinomial(probabilities, num_samples=1).item()
        # print("nt: "+str(next_token))
        if next_token == end_token:
            break
        start_text_token.append(next_token)
        tokens = torch.cat((tokens, torch.tensor([[next_token]], device=device)), dim=0)
    generated_name = enc.decode(start_text_token)
    return generated_name
    
# Generate 10 names
for _ in range(20):
    name = sample(model, enc, 's')
    print(name)
print('---------------------------')
val_loss = validation()
print(val_loss)
n_params = sum(p.numel() for p in model.parameters())
print(n_params)
# 2.6517
