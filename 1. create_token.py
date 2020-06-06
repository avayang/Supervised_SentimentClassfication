import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import time
import os
import re
from itertools import chain
from transformers import BertTokenizer
PRETRAINED_MODEL_NAME = "bert-base-uncased" #英文pretrain(不區分大小寫)
print(torch.__version__)
#1.3.1

# load tokenizer
tokenizer = BertTokenizer.from_pretrained(PRETRAINED_MODEL_NAME)
vocab = tokenizer.vocab
print("Dict size:", len(vocab))

# see some random tokens and indexes mapping
import random
rand_token = random.sample(list(vocab), 10)
rand_index = [vocab[i] for i in rand_token]

random_mapping = {}
random_mapping["Token"] = rand_token
random_mapping["Index"] = rand_index
pd.DataFrame(random_mapping)
