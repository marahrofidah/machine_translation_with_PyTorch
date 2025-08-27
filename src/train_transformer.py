import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import Adam, AdamW
import time
import math
from tqdm import tqdm
import os

# Import model Transformer - GANTI JADI ABSOLUTE IMPORT
from transformer_model import TransformerModel  # Hapus titik di awal
from data_loader import get_data_loaders
from utils import save_checkpoint

def train_transformer(config):
    """Main training function for Transformer model"""
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load data menggunakan fungsi yang sudah ada
    train_loader, val_loader, test_loader, src_sp_model, tgt_sp_model = get_data_loaders(
        batch_size=config['batch_size']
    )
    
    # Dapatkan ukuran vocab dari tokenizer
    src_vocab_size = src_sp_model.get_piece_size()
    tgt_vocab_size = tgt_sp_model.get_piece_size()
    
    # Initialize model
    model = TransformerModel(
        src_vocab_size=src_vocab_size,
        tgt_vocab_size=tgt_vocab_size,
        d_model=config['d_model'],
        nhead=config['nhead'],
        num_encoder_layers=config['num_encoder_layers'],
        num_decoder_layers=config['num_decoder_layers'],
        dim_feedforward=config['dim_feedforward'],
        dropout=config['dropout'],
        max_seq_length=config['max_seq_length']
    ).to(device)
    
    # Optimizer
    optimizer = AdamW(model.parameters(), lr=config['learning_rate'], 
                     weight_decay=config['weight_decay'])
    
    # Loss function - PERBAIKI PAD_ID
    pad_id = src_sp_model.pad_id() if hasattr(src_sp_model, 'pad_id') else 0
    criterion = nn.CrossEntropyLoss(ignore_index=pad_id)
    
    # Training loop
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    
    for epoch in range(config['num_epochs']):
        start_time = time.time()
        
        # Train
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device, config, epoch, pad_id)
        
        # Validate
        val_loss = validate_epoch(model, val_loader, criterion, device, config, pad_id)
        
        # Save losses
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            # Pastikan folder models ada
            os.makedirs("models", exist_ok=True)
            save_checkpoint({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, "models/best_transformer_model.pth")
        
        # Print progress
        epoch_mins, epoch_secs = divmod(time.time() - start_time, 60)
        print(f'Epoch: {epoch+1:03d} | Time: {epoch_mins:.0f}m {epoch_secs:.0f}s')
        print(f'\tTrain Loss: {train_loss:.4f} | Train PPL: {math.exp(train_loss):.4f}')
        print(f'\tVal Loss: {val_loss:.4f} | Val PPL: {math.exp(val_loss):.4f}')
    
    return model, train_losses, val_losses

def train_epoch(model, data_loader, optimizer, criterion, device, config, epoch, pad_id):
    """One training epoch"""
    model.train()
    total_loss = 0
    
    progress_bar = tqdm(data_loader, desc=f'Epoch {epoch+1}')
    
    for batch_idx, batch in enumerate(progress_bar):
        src = batch['src_ids'].to(device)
        tgt = batch['tgt_ids'].to(device)
        
        # Prepare input and output for teacher forcing
        tgt_input = tgt[:, :-1]
        tgt_output = tgt[:, 1:]
        
        # Create masks
        tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
        src_padding_mask = (src == pad_id)
        tgt_padding_mask = (tgt_input == pad_id)
        
        # Forward pass
        optimizer.zero_grad()
        output = model(src, tgt_input, tgt_mask=tgt_mask,
                      src_key_padding_mask=src_padding_mask,
                      tgt_key_padding_mask=tgt_padding_mask)
        
        # Calculate loss
        output = output.contiguous().view(-1, output.size(-1))
        tgt_output = tgt_output.contiguous().view(-1)
        loss = criterion(output, tgt_output)
        
        # Backward pass
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), config['clip_value'])
        optimizer.step()
        
        total_loss += loss.item()
        progress_bar.set_postfix({'loss': loss.item()})
    
    return total_loss / len(data_loader)

def validate_epoch(model, data_loader, criterion, device, config, pad_id):
    """Validation epoch"""
    model.eval()
    total_loss = 0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc='Validation'):
            src = batch['src_ids'].to(device)
            tgt = batch['tgt_ids'].to(device)
            
            # Prepare input and output
            tgt_input = tgt[:, :-1]
            tgt_output = tgt[:, 1:]
            
            # Create masks
            tgt_mask = model.generate_square_subsequent_mask(tgt_input.size(1)).to(device)
            src_padding_mask = (src == pad_id)
            tgt_padding_mask = (tgt_input == pad_id)
            
            # Forward pass
            output = model(src, tgt_input, tgt_mask=tgt_mask,
                          src_key_padding_mask=src_padding_mask,
                          tgt_key_padding_mask=tgt_padding_mask)
            
            # Calculate loss
            output = output.contiguous().view(-1, output.size(-1))
            tgt_output = tgt_output.contiguous().view(-1)
            loss = criterion(output, tgt_output)
            
            total_loss += loss.item()
    
    return total_loss / len(data_loader)
# gataula pusing
if __name__ == "__main__":
    # Konfigurasi training
    transformer_config = {
        'd_model': 512,
        'nhead': 8,
        'num_encoder_layers': 3,
        'num_decoder_layers': 3,
        'dim_feedforward': 2048,
        'dropout': 0.1,
        'max_seq_length': 100,
        'batch_size': 32,
        'num_epochs': 20,
        'learning_rate': 0.0001,
        'weight_decay': 0.0001,
        'clip_value': 1.0
    }
    
    print("Starting Transformer training...")
    model, train_losses, val_losses = train_transformer(transformer_config)
    print("Transformer training completed!")