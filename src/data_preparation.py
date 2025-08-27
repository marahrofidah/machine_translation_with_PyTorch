import pandas as pd
import sentencepiece as spm
from sklearn.model_selection import train_test_split
import os
import re
from tqdm import tqdm

def clean_text(text):
    """Membersihkan dan normalisasi teks"""
    # Convert to lowercase
    text = text.lower()
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text)
    # Add space around punctuation
    text = re.sub(r'([.!?,;:])', r' \1 ', text)
    # Remove special characters (keep alphanumeric and basic punctuation)
    text = re.sub(r'[^a-zA-Z0-9.!?,;:]+', ' ', text)
    # Clean up spaces
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def load_and_clean_data(file_path, max_samples=50000):
    """Load dan bersihkan dataset"""
    print("Loading and cleaning data...")
    
    # Cek jika file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File tidak ditemukan: {file_path}")
    
    # Baca dataset line by line untuk handle format yang berbeda
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    print(f"Total lines read: {len(lines)}")
    
    for i, line in enumerate(lines[:max_samples]):
        parts = line.strip().split('\t')
        if len(parts) >= 2:
            data.append({
                'english': clean_text(parts[0]),
                'indonesian': clean_text(parts[1])
            })
        elif len(parts) == 1:
            # Handle case dimana mungkin hanya ada 1 kolom
            print(f"Warning: Line {i+1} hanya memiliki 1 kolom: {line.strip()}")
    
    df = pd.DataFrame(data)
    
    if len(df) == 0:
        raise ValueError("Tidak ada data yang berhasil dibaca. Cek format file.")
    
    # Filter kalimat yang terlalu pendek/panjang
    df = df[df['english'].str.split().str.len() <= 50]
    df = df[df['indonesian'].str.split().str.len() <= 50]
    df = df[df['english'].str.split().str.len() >= 3]
    df = df[df['indonesian'].str.split().str.len() >= 3]
    
    print(f"Total samples after cleaning: {len(df)}")
    return df

def train_sentencepiece_model(texts, model_prefix, vocab_size=8000):
    """Train SentencePiece model"""
    # Pastikan directory untuk model exists
    model_dir = os.path.dirname(model_prefix)
    if model_dir and not os.path.exists(model_dir):
        os.makedirs(model_dir, exist_ok=True)
    
    # Save texts to temporary file di current directory
    temp_file = f"temp_{os.path.basename(model_prefix)}.txt"
    with open(temp_file, 'w', encoding='utf-8') as f:
        for text in texts:
            f.write(text + '\n')
    
    # Train SentencePiece model
    spm.SentencePieceTrainer.train(
        input=temp_file,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        character_coverage=1.0,
        model_type='bpe',
        pad_id=0,
        unk_id=1,
        bos_id=2,
        eos_id=3,
        pad_piece='<pad>',
        unk_piece='<unk>',
        bos_piece='<s>',
        eos_piece='</s>'
    )
    
    # Clean up temp file
    os.remove(temp_file)
    
    # Load model to verify
    sp_model = spm.SentencePieceProcessor()
    sp_model.load(f"{model_prefix}.model")
    print(f"{model_prefix} vocab size: {sp_model.get_piece_size()}")
    return sp_model

def split_data(df, test_size=0.2, val_size=0.1):
    """Split data into train, validation, and test sets"""
    # First split: train + temp, then split temp into validation + test
    train_df, temp_df = train_test_split(df, test_size=test_size + val_size, random_state=42)
    val_df, test_df = train_test_split(temp_df, test_size=test_size/(test_size + val_size), random_state=42)
    
    print(f"Train: {len(train_df)}, Validation: {len(val_df)}, Test: {len(test_df)}")
    return train_df, val_df, test_df

def save_data(df, filename):
    """Save data to file"""
    # Pastikan directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df[['english', 'indonesian']].to_csv(filename, sep='\t', index=False, header=False)
    print(f"Saved {len(df)} samples to {filename}")

def main():
    # Config - PASTIKAN PATH INI BENAR
    DATA_PATH = "data/raw/ind.txt"
    MAX_SAMPLES = 50000
    VOCAB_SIZE = 8000
    OUTPUT_DIR = "data/processed"
    TOKENIZER_DIR = "data/tokenizer"
    
    # Pastikan semua directory exists
    os.makedirs("data/raw", exist_ok=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(TOKENIZER_DIR, exist_ok=True)
    
    # Debug: Print current directory and check if file exists
    print(f"Current working directory: {os.getcwd()}")
    print(f"Data path: {DATA_PATH}")
    print(f"File exists: {os.path.exists(DATA_PATH)}")
    
    # 1. Load and clean data
    df = load_and_clean_data(DATA_PATH, MAX_SAMPLES)
    
    # 2. Split data
    train_df, val_df, test_df = split_data(df)
    
    # 3. Save split data
    save_data(train_df, os.path.join(OUTPUT_DIR, "train.tsv"))
    save_data(val_df, os.path.join(OUTPUT_DIR, "val.tsv"))
    save_data(test_df, os.path.join(OUTPUT_DIR, "test.tsv"))
    
    # 4. Train SentencePiece models for both languages
    print("\nTraining English SentencePiece model...")
    all_english_texts = list(train_df['english']) + list(val_df['english'])
    eng_sp_model = train_sentencepiece_model(all_english_texts, os.path.join(TOKENIZER_DIR, "eng_sp"), VOCAB_SIZE)
    
    print("\nTraining Indonesian SentencePiece model...")
    all_indonesian_texts = list(train_df['indonesian']) + list(val_df['indonesian'])
    ind_sp_model = train_sentencepiece_model(all_indonesian_texts, os.path.join(TOKENIZER_DIR, "ind_sp"), VOCAB_SIZE)
    
    # 5. Test tokenization
    print("\nTesting tokenization:")
    test_eng = "hello how are you today?"
    test_ind = "halo apa kabar hari ini?"
    
    print(f"English: '{test_eng}'")
    print(f"Tokenized: {eng_sp_model.encode_as_pieces(test_eng)}")
    print(f"IDs: {eng_sp_model.encode_as_ids(test_eng)}")
    
    print(f"Indonesian: '{test_ind}'")
    print(f"Tokenized: {ind_sp_model.encode_as_pieces(test_ind)}")
    print(f"IDs: {ind_sp_model.encode_as_ids(test_ind)}")
    
    print("\nData preparation completed successfully!")

if __name__ == "__main__":
    main()