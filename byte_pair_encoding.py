# import pickle
# from collections import deque
# import os
# 
# import time
# 
# target_vocab = 80
# 
# encodings = {chr(i).encode("utf-16-le"): i for i in range(127)}
# 
# window = deque(maxlen=2)
# 
# CHUNK_SIZE = 65536
# 
# with open('Shakespeare.txt','rb') as src, open('src.tmp','wb') as dst:
#     for line in src:
#         for word in line:
#             byte_pair = word.to_bytes(2,byteorder="little")
#             dst.write(byte_pair)
# 
# j = 0
# while j < target_vocab:
#     t1 = time.time()
#     pairs = {}
#     window.clear()
#     with open('src.tmp','rb') as file:
#         while True:
#             chunk = file.read(CHUNK_SIZE)
#             if not chunk:
#                 break
#             for i in range(0,len(chunk),2):
#                 byte_pair = chunk[i:i+2]
#                 window.append(byte_pair)
#                 if len(window) == 2:
#                     pair = (window[0],window[1])
#                     if pair not in pairs:
#                         pairs[pair] = 1
#                     else:
#                         pairs[pair] += 1
#     j += 1
#     max_key = max(pairs,key=pairs.get)
#     encodings[max_key] = 126 + j
#     
#     with open('src.tmp','rb') as src, open('dst.tmp','wb') as dst:
#         tail = b''
#         while True:
#             chunk = src.read(CHUNK_SIZE)
#             if not chunk:
#                 if window:
#                     dst.write(tail)
#                 break
#             
#             data = tail + chunk
#             process_part = data
#             
#             
#             window.clear()
#             for i in range(0,len(process_part),2):
#                 byte_pair = process_part[i:i+2]
#                 window.append(byte_pair)
#                 if len(window) == 2:
#                     if (window[0],window[1]) in encodings:
#                         num = encodings[(window[0],window[1])]
#                         dst.write(num.to_bytes(2, byteorder="little"))
#                         window.clear()
#                     else:
#                         dst.write(window[0])
#             if window:
#                 tail = data[-2:]
#             else:
#                 tail = b''
#                 
#     os.remove('src.tmp')
#     os.replace('dst.tmp','src.tmp')
#     t2 = time.time()
#     print(t2-t1)
# 
# with open("Encodings.pkl", "wb") as f:
#     pickle.dump({v: k for k, v in encodings.items()}, f)
#         
#     
# 
# 
# print(encodings)

from tokenizers import Tokenizer
from tokenizers.trainers import BpeTrainer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel
from tokenizers.decoders import ByteLevel as ByteLevelDecoder

tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
tokenizer.decoder = ByteLevelDecoder()
trainer = BpeTrainer(vocab_size=2000)
tokenizer.train(['data/shakespeare_train.txt'],trainer)
tokenizer.save("data/tokenizer.json")