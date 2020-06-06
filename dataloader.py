from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence


# input "samples" is a list，every element is a sample from "create_dataset"
# each sample contains:
# - tokens_tensor
# - segments_tensor
# - label_tensor
# zero padding tokens_tensor and segments_tensor，and generate masks_tensors

def create_mini_batch(samples):
    """"
    4 tensors are needed：
    - tokens_tensors  : (batch_size, max_seq_len_in_batch)
    - segments_tensors: (batch_size, max_seq_len_in_batch)
    - masks_tensors   : (batch_size, max_seq_len_in_batch)
    - label_ids       : (batch_size)
    """

    tokens_tensors = [s[0] for s in samples]
    segments_tensors = [s[1] for s in samples]
    
    
    if samples[0][2] is not None: # for trainset
        label_ids = torch.stack([s[2] for s in samples])
    else: # for testset
        label_ids = None
    
    # zero padding 
    tokens_tensors = pad_sequence(tokens_tensors, batch_first=True)
    segments_tensors = pad_sequence(segments_tensors, batch_first=True)
    
    # attention masks，set 0 for zero padding positions, set 1 for actual token positions
    masks_tensors = torch.zeros(tokens_tensors.shape, dtype=torch.long)
    masks_tensors = masks_tensors.masked_fill(tokens_tensors != 0, 1)
    
    return tokens_tensors, segments_tensors, masks_tensors, label_ids

BATCH_SIZE = 32
trainloader = DataLoader(trainset, batch_size = BATCH_SIZE, collate_fn = create_mini_batch, shuffle=True)
valloader = DataLoader(valset, batch_size = BATCH_SIZE, collate_fn = create_mini_batch, shuffle=False)
testloader = DataLoader(testset, batch_size = BATCH_SIZE, collate_fn = create_mini_batch, shuffle=False)

data = next(iter(trainloader))
tokens_tensors, segments_tensors, masks_tensors, label_ids = data
print(tokens_tensors)
print(segments_tensors)
print(masks_tensors)
print(label_ids)
