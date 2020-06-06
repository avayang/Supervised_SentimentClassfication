from transformers import BertForSequenceClassification

NUM_LABELS = 2

model = BertForSequenceClassification.from_pretrained(
    PRETRAINED_MODEL_NAME, num_labels=NUM_LABELS)


print("""
name      module
--------------------
""")

for name, module in model.named_children():
    if name == "bert":
        for n, _ in module.named_children():
            print("{:10}{}".format(name,n) )
    else:
        print("{:10} {}".format(name, module))
