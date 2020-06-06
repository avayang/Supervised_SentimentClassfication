%%time
from sklearn.metrics import accuracy_score

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
print("device:",device)
model = model.to(device)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)
EPOCHS = 10

for epoch in range(EPOCHS):
    correct = 0
    #total = 0
    train_loss , val_loss = 0.0 , 0.0
    train_acc, val_acc = 0, 0
    n, m = 0, 0
    model.train()
    for data in trainloader:
        n += 1
        tokens_tensors, segments_tensors,masks_tensors, labels = [t.to(device) for t in data]
        
        optimizer.zero_grad()
        
        # forward pass
        outputs = model(input_ids=tokens_tensors, 
                        token_type_ids=segments_tensors, 
                        attention_mask=masks_tensors, 
                        labels=labels)
                        
        # outputs order: "(loss), logits, (hidden_states), (attentions)"
        
        loss = outputs[0]
        loss.backward()
        optimizer.step()
        
        #get prediction and calulate acc
        logits = outputs[1]
        _, pred = torch.max(logits.data, 1)
        train_acc += accuracy_score(pred.cpu().tolist() , labels.cpu().tolist())

        # track batch loss
        train_loss += loss.item()
    
    #validation
    with torch.no_grad():
        model.eval()
        for data in valloader:
            m += 1
            tokens_tensors, segments_tensors,masks_tensors, labels = [t.to(device) for t in data]
            val_outputs = model(input_ids=tokens_tensors, 
                        token_type_ids=segments_tensors, 
                        attention_mask=masks_tensors, 
                        labels=labels)
            
            logits = val_outputs[1]
            _, pred = torch.max(logits.data, 1)
            val_acc += accuracy_score(pred.cpu().tolist() , labels.cpu().tolist())
            val_loss += val_outputs[0].item()

    print("[epoch %d] loss: %.4f, acc: %.4f, val loss: %4f, val acc: %4f" %
          (epoch+1, train_loss/n, train_acc/n, val_loss/m,  val_acc/m  ))

print("Done")
