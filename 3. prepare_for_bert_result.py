# select a sample randomly
sample_idx = 10

# get tensors
tokens_tensor, segments_tensor, label_tensor, origin_text = trainset[sample_idx]

# revert tokens_tensors into tokens 
tokens = tokenizer.convert_ids_to_tokens(tokens_tensor.tolist())

print("token:\n", tokens,"\n")
print("origin_text:\n", origin_text,"\n")
print("label:", label_map[int(label_tensor.numpy())],"\n")
print("tokens_tensor:\n", tokens_tensor,"\n")
print("segment tensor:\n", segments_tensor, "\n")
print("label tensor:\n", label_tensor)
