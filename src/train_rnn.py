import torch
import torch.nn as nn
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import os
import math
from datetime import datetime

from model_rnn import Seq2Seq, EncoderRNN, DecoderRNN
from data_loader import get_data_loaders

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Initialize model
        self._init_model()
        
        # Initialize optimizer and loss function
        self.optimizer = Adam(self.model.parameters(), lr=config['lr'])
        self.criterion = nn.CrossEntropyLoss(ignore_index=config['tgt_pad_idx'])
        
        # Tensorboard writer
        self.writer = SummaryWriter(f"runs/{config['model_name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        
    def _init_model(self):
        # Load tokenizers to get vocab sizes
        from sentencepiece import SentencePieceProcessor
        src_sp = SentencePieceProcessor()
        src_sp.load('data/tokenizer/eng_sp.model')
        tgt_sp = SentencePieceProcessor()
        tgt_sp.load('data/tokenizer/ind_sp.model')
        
        # Create encoder and decoder
        encoder = EncoderRNN(
            vocab_size=src_sp.get_piece_size(),
            embed_size=self.config['embed_size'],
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout']
        )
        
        decoder = DecoderRNN(
            vocab_size=tgt_sp.get_piece_size(),
            embed_size=self.config['embed_size'],
            hidden_size=self.config['hidden_size'],
            num_layers=self.config['num_layers'],
            dropout=self.config['dropout']
        )
        
        self.model = Seq2Seq(
            encoder=encoder,
            decoder=decoder,
            src_pad_idx=src_sp.pad_id(),
            device=self.device
        ).to(self.device)
        
        self.config['src_pad_idx'] = src_sp.pad_id()
        self.config['tgt_pad_idx'] = tgt_sp.pad_id()
        
    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch}')
        
        for batch_idx, batch in enumerate(progress_bar):
            src = batch['src_ids'].to(self.device)
            tgt = batch['tgt_ids'].to(self.device)
            
            # Forward pass
            output = self.model(src, tgt)
            
            # Calculate loss
            output_dim = output.shape[-1]
            output = output[:, 1:].reshape(-1, output_dim)
            tgt = tgt[:, 1:].reshape(-1)
            
            loss = self.criterion(output, tgt)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['clip'])
            
            self.optimizer.step()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': loss.item()})
            
            # Log to tensorboard
            global_step = epoch * len(train_loader) + batch_idx
            self.writer.add_scalar('train/loss', loss.item(), global_step)
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader, epoch):
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in val_loader:
                src = batch['src_ids'].to(self.device)
                tgt = batch['tgt_ids'].to(self.device)
                
                output = self.model(src, tgt, teacher_forcing_ratio=0)
                
                output_dim = output.shape[-1]
                output = output[:, 1:].reshape(-1, output_dim)
                tgt = tgt[:, 1:].reshape(-1)
                
                loss = self.criterion(output, tgt)
                total_loss += loss.item()
        
        avg_loss = total_loss / len(val_loader)
        self.writer.add_scalar('val/loss', avg_loss, epoch)
        
        return avg_loss
    
    def train(self, train_loader, val_loader):
        best_val_loss = float('inf')
        
        for epoch in range(self.config['num_epochs']):
            train_loss = self.train_epoch(train_loader, epoch)
            val_loss = self.validate(val_loader, epoch)
            
            print(f'Epoch {epoch}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.save_model('best_model.pth')
            
            # Save checkpoint
            if epoch % self.config['save_interval'] == 0:
                self.save_model(f'checkpoint_epoch_{epoch}.pth')
    
    def save_model(self, filename):
        os.makedirs('models', exist_ok=True)
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'config': self.config
        }, f'models/{filename}')
    
    def load_model(self, filename):
        checkpoint = torch.load(f'models/{filename}', map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

def main():
    config = {
        'embed_size': 256,
        'hidden_size': 512,
        'num_layers': 2,
        'dropout': 0.3,
        'lr': 0.001,
        'batch_size': 32,
        'num_epochs': 10,
        'clip': 1.0,
        'save_interval': 2,
        'model_name': 'rnn_attention'
    }
    
    # Get data loaders
    train_loader, val_loader, _, _, _ = get_data_loaders(config['batch_size'])
    
    # Initialize trainer
    trainer = Trainer(config)
    
    # Start training
    trainer.train(train_loader, val_loader)

if __name__ == "__main__":
    main()
    # RNN GAJELASH