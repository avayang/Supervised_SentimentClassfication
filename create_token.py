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
