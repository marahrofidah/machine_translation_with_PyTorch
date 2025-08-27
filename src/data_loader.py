import torch
from torch.utils.data import Dataset, DataLoader
import sentencepiece as spm
import pandas as pd

class TranslationDataset(Dataset):
    def __init__(self, file_path, src_sp_model, tgt_sp_model):
        self.data = pd.read_csv(file_path, sep='\t', header=None, names=['src', 'tgt'])
        self.src_sp_model = src_sp_model
        self.tgt_sp_model = tgt_sp_model
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        src_text = self.data.iloc[idx]['src']
        tgt_text = self.data.iloc[idx]['tgt']
        
        # Tokenize source and target
        src_ids = self.src_sp_model.encode_as_ids(src_text)
        tgt_ids = self.tgt_sp_model.encode_as_ids(tgt_text)
        
        # Add special tokens
        src_ids = [self.src_sp_model.bos_id()] + src_ids + [self.src_sp_model.eos_id()]
        tgt_ids = [self.tgt_sp_model.bos_id()] + tgt_ids + [self.tgt_sp_model.eos_id()]
        
        return {
            'src_ids': torch.tensor(src_ids, dtype=torch.long),
            'tgt_ids': torch.tensor(tgt_ids, dtype=torch.long),
            'src_text': src_text,
            'tgt_text': tgt_text
        }

def collate_fn(batch, src_pad_id, tgt_pad_id):
    """Custom collate function to pad sequences"""
    src_ids = [item['src_ids'] for item in batch]
    tgt_ids = [item['tgt_ids'] for item in batch]
    src_texts = [item['src_text'] for item in batch]
    tgt_texts = [item['tgt_text'] for item in batch]
    
    # Pad sequences
    src_padded = torch.nn.utils.rnn.pad_sequence(src_ids, batch_first=True, padding_value=src_pad_id)
    tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_ids, batch_first=True, padding_value=tgt_pad_id)
    
    return {
        'src_ids': src_padded,
        'tgt_ids': tgt_padded,
        'src_texts': src_texts,
        'tgt_texts': tgt_texts
    }

def get_data_loaders(batch_size=32):
    """Get data loaders for train, validation, and test"""
    # Load tokenizer models
    src_sp_model = spm.SentencePieceProcessor()
    src_sp_model.load('data/tokenizer/eng_sp.model')
    
    tgt_sp_model = spm.SentencePieceProcessor()
    tgt_sp_model.load('data/tokenizer/ind_sp.model')
    
    # Create datasets
    train_dataset = TranslationDataset('data/processed/train.tsv', src_sp_model, tgt_sp_model)
    val_dataset = TranslationDataset('data/processed/val.tsv', src_sp_model, tgt_sp_model)
    test_dataset = TranslationDataset('data/processed/test.tsv', src_sp_model, tgt_sp_model)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        collate_fn=lambda x: collate_fn(x, src_sp_model.pad_id(), tgt_sp_model.pad_id())
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, src_sp_model.pad_id(), tgt_sp_model.pad_id())
    )
    
    test_loader = DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=lambda x: collate_fn(x, src_sp_model.pad_id(), tgt_sp_model.pad_id())
    )
    
    return train_loader, val_loader, test_loader, src_sp_model, tgt_sp_model

# Test function
def test_data_loader():
    train_loader, val_loader, test_loader, src_sp, tgt_sp = get_data_loaders(batch_size=2)
    
    print(f"Source vocab size: {src_sp.get_piece_size()}")
    print(f"Target vocab size: {tgt_sp.get_piece_size()}")
    
    # Test one batch
    batch = next(iter(train_loader))
    print(f"Batch source shape: {batch['src_ids'].shape}")
    print(f"Batch target shape: {batch['tgt_ids'].shape}")
    print(f"Source texts: {batch['src_texts'][:2]}")
    print(f"Target texts: {batch['tgt_texts'][:2]}")
    
    return True

if __name__ == "__main__":
    test_data_loader()