import data_preperation
import yaml
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
import train_model
import torch

def load_config(config_path = 'config.yaml'):
    with open(config_path, 'r') as config_file:
        config = yaml.safe_load(config_file)
    return config

def runner(config):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # data loading
    print("Loading and preprocessing started ...")
    ds = data_preperation.data_loading_and_preprocessing('master_data.csv')
    print("Loading and preprocessing completed")
    
    tokenizer = AutoTokenizer.from_pretrained('distilbert-base-uncased')
    
    train_loader, val_loader = data_preperation.process_data(ds, tokenizer, padding = 'max_length',
                                                             truncation = config['data_config']['truncation'], 
                                                             max_length = config['data_config']['max_seq_len'], 
                                                             train_batch_size = config['data_config']['train_batch_size'], 
                                                             val_batch_size = config['data_config']['val_batch_size'])
    
    # Load a model
    print("Loading the model...")
    model = AutoModelForSequenceClassification.from_pretrained('distilbert-base-uncased',
                                                               num_labels=config['data_config']['n_classes']).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)
    
    # training the model
    print("training the model")
    train_model.run_training(model, optimizer, train_loader, val_loader, 
                config['model_config']['n_epochs'], device)


if __name__ == "__main__":
    config = load_config()
    runner(config)