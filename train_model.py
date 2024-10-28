from tqdm import tqdm
from sklearn.metrics import accuracy_score
import torch
import numpy as np

def train_runner(model, optimizer, train_loader, device, epoch):
    model.train()
    
    losses = []
    predicted = []
    actuals = []
    batch_itr = tqdm(enumerate(train_loader), total=len(train_loader))
    for _,batch in batch_itr:
        complaint_ids = batch['complaint_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device)
        
        output = model(complaint_ids, attention_mask = attention_mask, labels = labels)
        loss = output.loss
        logits = output.logits
        pred = torch.argmax(logits, dim=-1)
        
        predicted.append(pred)
        actuals.append(labels)
        losses.append(loss)
        
        loss.backward()
        optimizer.step()  
        
        batch_itr.set_postfix({"train_loss":loss})
        # print(f"Epoch {epoch}, Loss: {loss.item()}")
    f_loss = np.mean(losses)
    return f_loss, accuracy_score(predicted, actuals)
        
def val_runner(model, train_loader, device, epoch):
    model.eval()
    predicted = []
    actuals = []
    losses = []
    with torch.no_grad():
        batch_itr = tqdm(enumerate(train_loader),total = len(train_loader))
        for _,batch in batch_itr:
            complaint_ids = batch['complaint_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
                        
            output = model(complaint_ids, attention_mask = attention_mask, labels = labels)
            loss = output.loss
            losses.append(loss)
            logits = output.logits
            pred = torch.argmax(logits, dim = -1)
            
            predicted.append(pred)
            actuals.append(labels)
            losses.append(loss)

            # print(f"Epoch {epoch}, Loss: {loss.item()}")
            batch_itr.set_postfix({"train_loss":loss})
    
    f_loss = np.mean(losses)

    return f_loss, accuracy_score(predicted, actuals)
        


def run_training(model, optimizer, train_loader, val_loader, n_epochs, device):
    for epochs in range(n_epochs):
        train_loss, train_acc = train_runner(model, optimizer, train_loader, device, epochs)
        val_loss, val_acc = val_runner(model, val_loader, device, epochs)
        print(f"train loss: {train_loss}, train acc: {train_acc}")
        print(f"val loss: {val_loss}, val acc: {val_acc}")