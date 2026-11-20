import Transformer
import torch
from torch import nn
from tokenizers import Tokenizer

START_FROM = 50

CONTEXT = 512
BATCH_SIZE = 100

dt = torch.float32

tokenizer = Tokenizer.from_file("data/tokenizer.json")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    
torch.manual_seed(70)

def token_batches(file, context):
    with open(file, 'r') as f:
        text = f.read()
    
    tokens = torch.tensor(tokenizer.encode(text).ids, dtype=torch.long)
    
    num_batches = (len(tokens) - 1) // context
    total_len = num_batches * context
    
    inputs = tokens[:total_len].view(num_batches, context).to(device)
    targets = tokens[1:total_len + 1].view(num_batches, context).to(device)
    
    return inputs, targets

def main():
    softmax = Transformer.SoftMax.apply
    INPUT,TARGET = token_batches('data/shakespeare_test.txt',CONTEXT)
    model = Transformer.Transformer().to(device)
    model.to(dtype=dt)
    if START_FROM > 0:
        model.load_state_dict(torch.load(f'data/weights/weights_epoch{START_FROM}.pth', map_location=device,weights_only=True))
    model.eval()
    with torch.inference_mode():
        idx = torch.randperm(INPUT.size(0))
        INPUT = INPUT[idx]
        TARGET = TARGET[idx]
        epoch_loss = 0
        k = 0
        for j in range(0,INPUT.size()[0],BATCH_SIZE):
            logits = model(INPUT[j:j+BATCH_SIZE])
            loss = torch.nn.functional.cross_entropy(logits.permute(0, 2, 1), TARGET[j:j+BATCH_SIZE])
            epoch_loss += loss.item()
            k += 1
        average_loss = epoch_loss/k
        print(f"average loss over validation set: {average_loss}\n")
    
if __name__ == "__main__":
    main()
