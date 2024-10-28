import torch.utils
from transformers import AutoTokenizer
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd


def data_loading_and_preprocessing(filepath:str):
    master_data = pd.read_csv(filepath)
    master_data.drop_duplicates(inplace=True)
    master_data['label'] = master_data['Product'].factorize()[0]
    master_data.dropna(inplace=True)
    
    return master_data
    
    

class ClassificationDataset(Dataset):
    def __init__(self,ds, tokenizer, padding, truncation:bool, max_length:int):
        self.ds = ds      
        self.tokenizer = tokenizer
        self.padding = padding
        self.truncation = truncation
        self.max_length = max_length
        
    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, idx):
        complaint = self.ds.iloc[idx]['Consumer complaint narrative']
        label = self.ds.iloc[idx]['label']
        
        tokenized_complaint = self.tokenizer.encode_plus(complaint,
                                                         padding = self.padding, add_special_tokens = True, 
                                                                      truncation = self.truncation,
                                                                      max_length = self.max_length,
                                                                      return_attention_mask = True,
                                                                      return_tensors = 'pt')
        
        complaint_ids =  tokenized_complaint['input_ids'].flatten()
        attention_mask = tokenized_complaint['attention_mask'].flatten()
        label = torch.tensor(label,dtype=torch.int64)        
        
        return {   
            'complaint_ids': complaint_ids,
            'attention_mask': attention_mask,
            'label': label
        }
        

def process_data(ds, tokenizer, padding, truncation, max_length,
                 train_batch_size:int, val_batch_size:int):
       
    dataset = ClassificationDataset(ds, tokenizer, padding, truncation, max_length)
    
    train_size = int(len(dataset)*0.8)
    val_size = len(dataset) - train_size

    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
    train_loader = DataLoader(train_dataset, shuffle=True, batch_size=train_batch_size)
    val_loader = DataLoader(val_dataset, shuffle=False, batch_size=val_batch_size)
    
    return train_loader, val_loader