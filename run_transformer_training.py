import sys
import os

# Tambahkan path src ke Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from train_transformer import train_transformer

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