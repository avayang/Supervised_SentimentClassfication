def preprocess(sen):
    
    # remove html tags
    sentence = re.sub(r'<[^>]+>', '', sen)
    # remove multiple spaces
    sentence = re.sub(r'\s+', ' ', sentence)
    # remove punctuations and numbers
    sentence = re.sub('[^a-zA-Z]', ' ', sentence)
    # Single character removal
    sentence = re.sub(r'\s+[a-zA-Z]\s+', ' ', sentence)
    
    return sentence


def readimdb(mode):
    """
    How return value looks like:
    [[text1, 1 or 0], [text2, 1 or 0], ...]
    - 1:"positive"
    - 0:"negative"
    """
    
    data = []
    df = pd.read_csv(mode + ".tsv", sep="\t")
    #SAMPLE_FRAC = 0.01
    #df = df.sample(frac=SAMPLE_FRAC, random_state=10)
    
    for i in range(len(df)):
        if df.iloc[i, 2] == "positive":
            data.append([preprocess(df.iloc[i, 0]), 1])
        else:
            data.append([preprocess(df.iloc[i, 0]), 0])
                    
    return data

label_map = {0: "negative", 1: "positive"}

# create dataset
class create_dataset(Dataset):
    """
    Transform every review entry into the form compatible with BERT, and return 3 tensors:
    - tokens_tensor：index for tokens，including "[CLS]" and "[SEP]"
    - segments_tensor：the input is one single review, no segments, set segments_tensor = 1
    - label_tensor：index for label, if it's testset then return to None
   """
    
    def __init__(self, mode, tokenizer):
        
        assert mode in ["train", "test"]  
        self.mode = mode
        self.df = readimdb(self.mode)
        self.len = len(self.df)
        self.maxlen = 50
        self.tokenizer = tokenizer  # BERT tokenizer
        
    def __getitem__(self, idx):
        
        if self.mode == "test":
            text = self.df[idx][0]
            label_tensor = None
        else:
            text = self.df[idx][0]
            label = self.df[idx][1]
            label_tensor = torch.tensor(label)
           
        
        # BERT tokens for text review, and add [SEP]
        word_pieces = ["[CLS]"]
        tokens = self.tokenizer.tokenize(text)
        word_pieces += tokens[:self.maxlen] + ["[SEP]"]
        len_text = len(word_pieces)
        

        # convert tokens into index
        ids = self.tokenizer.convert_tokens_to_ids(word_pieces)
        tokens_tensor = torch.tensor(ids)
        
        # set segments_tensor = 1
        segments_tensor = torch.tensor([1] * len_text, dtype=torch.long)
        
        return (tokens_tensor, segments_tensor, label_tensor, text)
    
    
    def __len__(self):
        return self.len
    
# initialize dataset   
trainset = create_dataset("train", tokenizer = tokenizer)
testset = create_dataset("test", tokenizer = tokenizer)

val_size = int(trainset.__len__()*0.20) # set 20% of trainset as validation set
trainset, valset = random_split(trainset,[trainset.__len__()-val_size,val_size])
print("trainset size:", trainset.__len__())
print("valset size:", valset.__len__())
print("testset size:", testset.__len__())
