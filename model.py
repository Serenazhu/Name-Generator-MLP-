import torch
from torch.nn import functional as F
import tiktoken
import random

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
n1 = int(len(all_tokens) * 0.9)
# n2 = int(len(all_tokens) * 0.9)
train_tokens = all_tokens[:n1]
val_tokens = all_tokens[n1:]
# test_tokens = all_tokens[n2:]
# Define vocabulary size
vocab_size = enc.n_vocab # 50257
# import sys; sys.exit(0)
# Create dataset from tokens
def create_ds(tokens):
    xs = []
    ys = []
    for token in tokens:
        for ch1, ch2, ch3 in zip(token, token[1:], token[2:]):
            xs.append(ch1); xs.append(ch2)
            ys.append(ch3)
    xs = torch.tensor(xs, device=device).view(-1, 2) # (# of examples , 2)
    ys = torch.tensor(ys, device=device)
    return xs, ys

# Define the NLP model
class NLP(torch.nn.Module):
    def __init__(self, vocab_size, embed_size=32):
        super().__init__()
        self.embd = torch.nn.Embedding(vocab_size, embed_size)
        self.layer1 = torch.nn.Linear(embed_size*2, 32) # *2 cause the input is 2 chrs
        self.bn1 = torch.nn.BatchNorm1d(32) # https://arxiv.org/pdf/1502.03167
        self.layer2 = torch.nn.Linear(32, vocab_size) 
        self.dropout = torch.nn.Dropout(0.8)  # https://www.cs.toronto.edu/~rsalakhu/papers/srivastava14a.pdf

    def forward(self, x):
        x = self.embd(x) # (64, 2, 32)
        x = x.view(x.size(0), -1)  # Flatten the embeddings (64, 64)
        h1 = F.gelu(self.bn1(self.layer1(x))) #https://www.google.com/search?client=safari&sca_esv=6fc9685705fbb62e&sca_upv=1&rls=en&sxsrf=ADLYWIL8l5rfmc0NQno-f7Y6o3lCmvIy2w:1724801515858&q=gelu+activation+function&udm=2&fbs=AEQNm0Aa4sjWe7Rqy32pFwRj0UkWd8nbOJfsBGGB5IQQO6L3J_86uWOeqwdnV0yaSF-x2jogM63VUdBhAMVqo6r6ESHk5gYCycVYeSiTstipcfTqmJHGWyl0uJVRFOSrQ_UWiyXMTrB3DGLhC5G-ACymhRgfyx27lHEh0jX7vZV6vnVgydt5aBS3nKmaNpBcFguTv_Msr87RwC5WXgA1WW1iyf1J9ZVSmQ&sa=X&ved=2ahUKEwivtaStqpaIAxVFmO4BHV-RNIwQtKgLegQIFRAB&biw=1470&bih=840&dpr=2#vhid=U87bwdk7EFNAZM&vssid=mosaic
        h2 = self.layer2(h1)
        return h2

# Create dataset
Xtr, Ytr = create_ds(train_tokens)
Xval, Yval = create_ds(val_tokens)
Xte, Yte = create_ds(test_tokens)
# Initialize model
model = NLP(vocab_size).to(device)

# Training parameters
learning_rate = 1e-3  
num_epochs = 40
batch_size = 64  
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # https://arxiv.org/pdf/1412.6980
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.8) # Decreases the lr by a factor of gamma(20%) every step_size epochs.
# Training loop
for epoch in range(num_epochs):
    for i in range(0, 5000, batch_size):
        Xbatch = Xtr[i:i+batch_size].to(device)
        Ybatch = Ytr[i:i+batch_size].to(device)

        # Forward Pass
        logits = model(Xbatch)

        loss = F.cross_entropy(logits, Ybatch)
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # clipping the graidents that are bigger than 1 (returns the gradients before they are clipped.) 
        optimizer.step()
    print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}, Norm:{norm:.4f}')
    scheduler.step()
    current_lr = scheduler.get_last_lr()[0] # retrieves the most recently updated learning rate 
print('finished training!')

def validation():
    model.eval()
    total_loss = 0.0
    num_batch = 0
    with torch.no_grad():
        for i in range(0, len(Xval), batch_size):
            Xbatch = Xval[i:i+batch_size].to(device)
            Ybatch = Yval[i:i+batch_size].to(device)
            # Forward Pass
            logits = model(Xbatch)
            loss = F.cross_entropy(logits, Ybatch) #https://media.licdn.com/dms/image/v2/D4D12AQG8MVZausQXRQ/article-inline_image-shrink_400_744/article-inline_image-shrink_400_744/0/1662488155761?e=1729123200&v=beta&t=XYO8QYqZ4kgZocwkXEDMWgRO80BmROUjwp_UBVVjd0w
            total_loss += loss
            num_batch += 1
    averaged_loss = total_loss / num_batch
    return averaged_loss

def sample(model, enc, start_text, max_length=50, device='mps'):
    model.eval()
    input_gen = '<|start|>' + start_text
    tokens = enc.encode(input_gen, allowed_special={'<|start|>', '<|end|>'}) 
    tokens = torch.tensor(tokens, device=device).view(-1, 1) # ( , 1)
    start_text_token = enc.encode(start_text) 
    end_token = enc.encode('<|end|>', allowed_special={'<|start|>', '<|end|>'})[0]

    for _ in range(max_length):
        input_tokens = tokens[-2:].unsqueeze(0) # select the last two elements from the tokens tensor & adding a new dim (batch dim)
        logits = model(input_tokens)
        probabilities = F.softmax(logits, dim=1)
        next_token = torch.multinomial(probabilities, num_samples=1).item() #returns the token with the highest probability
        if next_token == end_token:
            break
        else:
            start_text_token.append(next_token)
            tokens = torch.cat((tokens, torch.tensor([[next_token]], device=device)), dim=0)
    generated_name = enc.decode(start_text_token)
    return generated_name
    
# Generate 10 names
for _ in range(20):
    name = sample(model, enc, 'S')
    print(name)
print('---------------------------')
val_loss = validation()
print("Validation Loss: ")
print(val_loss)
n_params = sum(p.numel() for p in model.parameters())
print(n_params)
