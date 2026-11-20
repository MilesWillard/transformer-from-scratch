import Transformer
import torch
from torch import nn
from tokenizers import Tokenizer

START_FROM = 50

CONTEXT = 512

dt = torch.float32

tokenizer = Tokenizer.from_file("data/tokenizer.json")

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")
    
#torch.manual_seed(2)

def get_input():
    text = input('Enter text: ')
    tokens = torch.tensor(tokenizer.encode(text).ids,device=device)
    len_input = len(tokens)
    return (tokens,len_input)

def main():
    TEMPERATURE = .8
    softmax = Transformer.SoftMax.apply
    model = Transformer.Transformer().to(device)
    model.to(dtype=dt)
    model.load_state_dict(torch.load(f'data/weights/weights_epoch{START_FROM}.pth', map_location=device,weights_only=True))
    model.eval()
    INPUT, word = get_input()
    INPUT = INPUT.unsqueeze(0)
    sentence = ''
    with torch.inference_mode():
        for i in range(CONTEXT-word):
            OUTPUT = softmax(model(INPUT)/TEMPERATURE).squeeze()
            t = torch.multinomial(OUTPUT,1)
            next_word = torch.multinomial(OUTPUT,1)[-1]
            INPUT = torch.cat((INPUT,next_word.view(1,1)),dim=1)
            sentence += tokenizer.decode([next_word.item()])
    print(sentence)
    
if __name__ == "__main__":
    main()
