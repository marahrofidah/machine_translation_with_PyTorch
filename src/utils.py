import torch
import os
import sentencepiece as spm

def translate_sentence(sentence, model, src_sp_model, tgt_sp_model, device, max_length=50):
    """Translate a single sentence"""
    model.eval()
    
    # Tokenize source sentence
    src_ids = src_sp_model.encode_as_ids(sentence)
    src_ids = [src_sp_model.bos_id()] + src_ids + [src_sp_model.eos_id()]
    src_tensor = torch.LongTensor(src_ids).unsqueeze(0).to(device)
    
    # Create mask
    mask = (src_tensor != src_sp_model.pad_id()).unsqueeze(1).to(device)
    
    # Encoder forward
    with torch.no_grad():
        encoder_outputs, hidden = model.encoder(src_tensor)
    
    # Decoder forward
    tgt_ids = [tgt_sp_model.bos_id()]
    
    for _ in range(max_length):
        tgt_tensor = torch.LongTensor([tgt_ids[-1]]).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output, hidden, _ = model.decoder(tgt_tensor, hidden, encoder_outputs, mask)
        
        pred_token = output.argmax(1).item()
        tgt_ids.append(pred_token)
        
        if pred_token == tgt_sp_model.eos_id():
            break
    
    # Decode target tokens
    translated = tgt_sp_model.decode_ids(tgt_ids[1:-1])  # Remove <s> and </s>
    
    return translated

def save_vocab_info(src_sp_model, tgt_sp_model, filepath):
    """Save vocabulary information"""
    with open(filepath, 'w', encoding='utf-8') as f:
        f.write("Source Vocabulary:\n")
        for i in range(src_sp_model.get_piece_size()):
            f.write(f"{i}: {src_sp_model.id_to_piece(i)}\n")
        
        f.write("\nTarget Vocabulary:\n")
        for i in range(tgt_sp_model.get_piece_size()):
            f.write(f"{i}: {tgt_sp_model.id_to_piece(i)}\n")

def save_checkpoint(state, filename):
    """Save model checkpoint"""
    # Create directory if it doesn't exist
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
    print(f"Checkpoint saved to {filename}")

def load_checkpoint(model, optimizer, filename):
    """Load model checkpoint"""
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['train_loss'], checkpoint['val_loss']

# Tambahkan fungsi untuk generate mask jika belum ada
def generate_src_mask(src, pad_idx=0):
    """Generate source mask for padding"""
    return (src != pad_idx).unsqueeze(1).unsqueeze(2)

def generate_tgt_mask(tgt, pad_idx=0):
    """Generate target mask for padding and causal masking"""
    tgt_mask = (tgt != pad_idx).unsqueeze(1).unsqueeze(2)
    seq_length = tgt.size(1)
    nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()
    return tgt_mask & nopeak_mask