import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
import sacrebleu
import numpy as np
import json
import time
from collections import defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
import sentencepiece as spm
import os

# Import model classes (sesuaikan dengan struktur project Anda)
from model_rnn import Seq2Seq as Seq2SeqWithAttention, EncoderRNN, DecoderRNN
from transformer_model import TransformerModel
from data_loader import TranslationDataset, get_data_loaders
import utils

class ModelEvaluator:
    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        self.device = device
        
    def load_model_and_vocab(self, model_path, model_type='transformer'):
        """Load trained model and SentencePiece tokenizers with auto-detect parameters"""
        print(f"Loading checkpoint from {model_path}...")
        
        try:
            checkpoint = torch.load(model_path, map_location=self.device)
            print(f"✅ Checkpoint loaded successfully")
        except Exception as e:
            print(f"❌ Error loading checkpoint: {e}")
            return None, None, None
        
        # Load SentencePiece tokenizers
        try:
            src_sp_model = spm.SentencePieceProcessor()
            src_sp_model.load('data/tokenizer/eng_sp.model')
            
            tgt_sp_model = spm.SentencePieceProcessor()
            tgt_sp_model.load('data/tokenizer/ind_sp.model')
            
            # Get vocab sizes from tokenizers
            src_vocab_size = src_sp_model.get_piece_size()
            tgt_vocab_size = tgt_sp_model.get_piece_size()
            
            print(f"✅ Tokenizers loaded: src={src_vocab_size}, tgt={tgt_vocab_size}")
        except Exception as e:
            print(f"❌ Error loading tokenizers: {e}")
            return None, None, None
        
        # Auto-detect model parameters from checkpoint
        if model_type == 'transformer':
            try:
                # Detect layers from checkpoint
                encoder_layers = []
                decoder_layers = []
                
                for key in checkpoint['model_state_dict'].keys():
                    if 'transformer.encoder.layers.' in key:
                        layer_num = int(key.split('layers.')[1].split('.')[0])
                        if layer_num not in encoder_layers:
                            encoder_layers.append(layer_num)
                    elif 'transformer.decoder.layers.' in key:
                        layer_num = int(key.split('layers.')[1].split('.')[0])
                        if layer_num not in decoder_layers:
                            decoder_layers.append(layer_num)
                
                num_encoder_layers = len(encoder_layers) if encoder_layers else 3
                num_decoder_layers = len(decoder_layers) if decoder_layers else 3
                
                print(f"Auto-detected: {num_encoder_layers} encoder layers, {num_decoder_layers} decoder layers")
                
                model = TransformerModel(
                    src_vocab_size=src_vocab_size,
                    tgt_vocab_size=tgt_vocab_size,
                    d_model=checkpoint.get('d_model', 512),
                    nhead=checkpoint.get('nhead', 8),
                    num_encoder_layers=num_encoder_layers,
                    num_decoder_layers=num_decoder_layers,
                    dim_feedforward=checkpoint.get('dim_feedforward', 2048),
                    dropout=checkpoint.get('dropout', 0.1),
                    max_seq_length=checkpoint.get('max_seq_length', 100)
                )
                
            except Exception as e:
                print(f"❌ Error creating transformer model: {e}")
                return None, None, None
                
        else:  # rnn
            try:
                # Auto-detect RNN parameters
                state_dict = checkpoint['model_state_dict']
                
                # Get config from checkpoint if available
                config = checkpoint.get('config', {})
                
                hidden_size = config.get('hidden_size', 512)
                embed_size = config.get('embed_size', 256)
                num_layers = config.get('num_layers', 2)
                dropout = config.get('dropout', 0.3)
                
                print(f"Auto-detected: hidden_size={hidden_size}, embed_size={embed_size}, num_layers={num_layers}")
                
                # Create encoder and decoder
                encoder = EncoderRNN(
                    vocab_size=src_vocab_size,
                    embed_size=embed_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout
                )
                
                decoder = DecoderRNN(
                    vocab_size=tgt_vocab_size,
                    embed_size=embed_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout
                )
                
                model = Seq2SeqWithAttention(
                    encoder=encoder,
                    decoder=decoder,
                    src_pad_idx=src_sp_model.pad_id(),
                    device=self.device
                )
                
            except Exception as e:
                print(f"❌ Error creating RNN model: {e}")
                return None, None, None
        
        # Load model weights with error handling
        try:
            # Try strict loading first
            model.load_state_dict(checkpoint['model_state_dict'], strict=True)
            print("✅ Model weights loaded successfully (strict)")
        except RuntimeError as e:
            try:
                # If strict fails, try non-strict
                model.load_state_dict(checkpoint['model_state_dict'], strict=False)
                print("✅ Model weights loaded successfully (non-strict)")
            except Exception as e2:
                print(f"❌ Loading failed: {e2}")
                return None, None, None
        
        model.to(self.device)
        model.eval()
        
        return model, src_sp_model, tgt_sp_model

    def translate_sentence(self, model, sentence_ids, src_sp_model, tgt_sp_model, max_length=100):
        """Translate a single sentence using SentencePiece tokenizers"""
        model.eval()
        
        # Convert to tensor if needed
        if isinstance(sentence_ids, list):
            src_tensor = torch.tensor(sentence_ids).unsqueeze(0).to(self.device)
        elif len(sentence_ids.shape) == 1:
            src_tensor = sentence_ids.unsqueeze(0).to(self.device)
        else:
            src_tensor = sentence_ids.to(self.device)
        
        with torch.no_grad():
            if hasattr(model, 'encoder') and hasattr(model, 'decoder'):  # RNN model
                # Encoder
                encoder_outputs, hidden = model.encoder(src_tensor)
                
                # Create mask with correct dimensions for attention
                mask = (src_tensor != src_sp_model.pad_id()).float()
                
                # Decoder - greedy decoding
                tgt_ids = [tgt_sp_model.bos_id()]
                
                for _ in range(max_length):
                    tgt_tensor = torch.tensor([tgt_ids[-1]]).to(self.device)
                    
                    output, hidden, _ = model.decoder(tgt_tensor, hidden, encoder_outputs, mask)
                    pred_token = output.argmax(1).item()
                    tgt_ids.append(pred_token)
                    
                    if pred_token == tgt_sp_model.eos_id():
                        break
                
                # Remove BOS and EOS if present
                if tgt_ids and tgt_ids[0] == tgt_sp_model.bos_id():
                    tgt_ids = tgt_ids[1:]
                if tgt_ids and tgt_ids[-1] == tgt_sp_model.eos_id():
                    tgt_ids = tgt_ids[:-1]
                    
                translated = tgt_sp_model.decode_ids(tgt_ids)
                
            else:  # Transformer model
                src_mask = (src_tensor != src_sp_model.pad_id())
                memory_key_padding_mask = ~src_mask
                
                tgt_ids = torch.tensor([[tgt_sp_model.bos_id()]], dtype=torch.long, device=self.device)
                
                for i in range(max_length - 1):
                    # Create target mask
                    tgt_mask = model.generate_square_subsequent_mask(tgt_ids.size(1)).to(self.device)
                    
                    output = model(src_tensor, tgt_ids,
                                   tgt_mask=tgt_mask,
                                   src_key_padding_mask=memory_key_padding_mask,
                                   tgt_key_padding_mask=(tgt_ids == tgt_sp_model.pad_id()))
                                   
                    pred_token = output[:, -1, :].argmax(dim=1)
                    
                    tgt_ids = torch.cat([tgt_ids, pred_token.unsqueeze(0)], dim=1)
                    
                    if pred_token.item() == tgt_sp_model.eos_id():
                        break
                
                # Remove BOS and EOS
                tgt_list = tgt_ids.squeeze().tolist()
                if isinstance(tgt_list, int):
                    tgt_list = [tgt_list]
                if tgt_list and tgt_list[0] == tgt_sp_model.bos_id():
                    tgt_list = tgt_list[1:]
                if tgt_list and tgt_list[-1] == tgt_sp_model.eos_id():
                    tgt_list = tgt_list[:-1]
                    
                translated = tgt_sp_model.decode_ids(tgt_list)

        return translated

    def evaluate_on_dataset(self, model, test_loader, src_sp_model, tgt_sp_model, beam_size=5, max_samples=None):
        """Evaluate model on test dataset using SentencePiece"""
        model.eval()
        
        predictions = []
        references = []
        
        print(f"Evaluating model...")
        
        total_samples = len(test_loader.dataset) if max_samples is None else min(max_samples, len(test_loader.dataset))
        processed = 0
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                if max_samples and processed >= max_samples:
                    break
                
                src_ids = batch['src_ids']
                tgt_ids = batch['tgt_ids']
                src_texts = batch['src_texts']
                tgt_texts = batch['tgt_texts']
                
                batch_size = src_ids.size(0)
                
                for i in range(batch_size):
                    if max_samples and processed >= max_samples:
                        break
                    
                    # Get source and reference
                    src_seq = src_ids[i]
                    reference = tgt_texts[i]
                    
                    if not reference.strip():  # Skip empty references
                        continue
                    
                    try:
                        # Generate prediction
                        prediction = self.translate_sentence(model, src_seq, src_sp_model, tgt_sp_model)
                        
                        predictions.append(prediction.strip())
                        references.append(reference.strip())
                        processed += 1
                        
                    except Exception as e:
                        print(f"Error translating sample {processed}: {e}")
                        continue
                
                if (batch_idx + 1) % 10 == 0:
                    print(f"Processed {processed}/{total_samples} samples...")
        
        return predictions, references
    
    def calculate_bleu(self, predictions, references):
        """Calculate BLEU score using sacrebleu"""
        filtered_pairs = [(p, r) for p, r in zip(predictions, references) if p.strip() and r.strip()]
        if not filtered_pairs:
            return 0.0
            
        preds, refs = zip(*filtered_pairs)
        refs_list = [[ref] for ref in refs]
        
        try:
            bleu = sacrebleu.corpus_bleu(preds, list(zip(*refs_list)))
            return bleu.score
        except:
            return 0.0
    
    def calculate_chrf(self, predictions, references):
        """Calculate chrF score"""
        filtered_pairs = [(p, r) for p, r in zip(predictions, references) if p.strip() and r.strip()]
        if not filtered_pairs:
            return 0.0
            
        preds, refs = zip(*filtered_pairs)
        refs_list = [[ref] for ref in refs]
        
        try:
            chrf = sacrebleu.corpus_chrf(preds, list(zip(*refs_list)))
            return chrf.score
        except:
            return 0.0

    def analyze_translations(self, predictions, references, src_sentences=None, num_examples=10):
        """Analyze translation examples with safe error handling"""
        print("\n" + "="*80)
        print("TRANSLATION EXAMPLES")
        print("="*80)
        
        # Handle None inputs
        if predictions is None:
            print("❌ Predictions is None - tidak ada data untuk dianalisis")
            return
            
        if references is None:
            print("❌ References is None - tidak ada data untuk dianalisis")
            return
        
        if len(predictions) == 0:
            print("❌ Predictions list kosong - tidak ada prediksi yang berhasil")
            return
            
        if len(references) == 0:
            print("❌ References list kosong - tidak ada referensi")
            return
        
        # Filter valid pairs
        valid_indices = []
        for i, (p, r) in enumerate(zip(predictions, references)):
            if p and r and isinstance(p, str) and isinstance(r, str):
                if p.strip() and r.strip():
                    valid_indices.append(i)
        
        if not valid_indices:
            print("❌ Tidak ada translation pairs yang valid")
            return
        
        print(f"✅ Found {len(valid_indices)} valid translation pairs dari {len(predictions)} total")
        
        # Show random examples
        import random
        random.seed(42)  # for reproducible results
        
        num_to_show = min(num_examples, len(valid_indices))
        selected_indices = random.sample(valid_indices, num_to_show)
        
        for i, idx in enumerate(selected_indices):
            print(f"\n📝 Example {i+1}:")
            if src_sentences and idx < len(src_sentences):
                print(f"Source:    {src_sentences[idx][:100]}...")
            print(f"Reference:  {references[idx][:100]}...")
            print(f"Prediction: {predictions[idx][:100]}...")
            print("-" * 60)
    
    def compare_models(self, rnn_model_path, transformer_model_path, test_loader, vocab_src, vocab_tgt, beam_size=5):
        """Compare RNN and Transformer models with robust error handling"""
        print("\n" + "="*80)
        print("MODEL COMPARISON")
        print("="*80)
        
        results = {}
        all_references = None
        
        # Evaluate RNN model
        print("Evaluating RNN + Attention model...")
        try:
            rnn_model, src_sp_model, tgt_sp_model = self.load_model_and_vocab(rnn_model_path, 'rnn')
            
            if rnn_model is None:
                print("❌ RNN model loading failed")
                results['rnn'] = {'bleu': 0, 'chrf': 0, 'time': 0, 'predictions': [], 'error': 'Model loading failed'}
            else:
                print("✅ RNN model loaded, starting evaluation...")
                start_time = time.time()
                
                rnn_predictions, references = self.evaluate_on_dataset(
                    rnn_model, test_loader, src_sp_model, tgt_sp_model, max_samples=1000
                )
                rnn_time = time.time() - start_time
                
                if rnn_predictions and references:
                    results['rnn'] = {
                        'bleu': self.calculate_bleu(rnn_predictions, references),
                        'chrf': self.calculate_chrf(rnn_predictions, references),
                        'time': rnn_time,
                        'predictions': rnn_predictions[:100],
                        'sample_count': len(rnn_predictions)
                    }
                    all_references = references
                    print(f"✅ RNN evaluation completed: BLEU {results['rnn']['bleu']:.2f}")
                else:
                    results['rnn'] = {'bleu': 0, 'chrf': 0, 'time': rnn_time, 'predictions': [], 'error': 'No valid predictions'}
                    print("❌ RNN evaluation failed: No valid predictions")
        
        except Exception as e:
            print(f"❌ RNN evaluation failed: {str(e)[:100]}...")
            results['rnn'] = {'bleu': 0, 'chrf': 0, 'time': 0, 'predictions': [], 'error': str(e)}
        
        # Evaluate Transformer model
        print("\nEvaluating Transformer model...")
        try:
            transformer_model, src_sp_model, tgt_sp_model = self.load_model_and_vocab(transformer_model_path, 'transformer')
            
            if transformer_model is None:
                print("❌ Transformer model loading failed")
                results['transformer'] = {'bleu': 0, 'chrf': 0, 'time': 0, 'predictions': [], 'error': 'Model loading failed'}
            else:
                print("✅ Transformer model loaded, starting evaluation...")
                start_time = time.time()
                
                transformer_predictions, references = self.evaluate_on_dataset(
                    transformer_model, test_loader, src_sp_model, tgt_sp_model, max_samples=1000
                )
                transformer_time = time.time() - start_time
                
                if transformer_predictions and references:
                    results['transformer'] = {
                        'bleu': self.calculate_bleu(transformer_predictions, references),
                        'chrf': self.calculate_chrf(transformer_predictions, references),
                        'time': transformer_time,
                        'predictions': transformer_predictions[:100],
                        'sample_count': len(transformer_predictions)
                    }
                    if all_references is None:
                        all_references = references
                    print(f"✅ Transformer evaluation completed: BLEU {results['transformer']['bleu']:.2f}")
                else:
                    results['transformer'] = {'bleu': 0, 'chrf': 0, 'time': transformer_time, 'predictions': [], 'error': 'No valid predictions'}
                    print("❌ Transformer evaluation failed: No valid predictions")
        
        except Exception as e:
            print(f"❌ Transformer evaluation failed: {str(e)[:100]}...")
            results['transformer'] = {'bleu': 0, 'chrf': 0, 'time': 0, 'predictions': [], 'error': str(e)}
        
        # Print comparison results
        print(f"\n📊 EVALUATION RESULTS:")
        print("="*50)
        
        successful_models = []
        for model_name, result in results.items():
            if 'error' not in result and result['bleu'] > 0:
                successful_models.append(model_name)
        
        if successful_models:
            print(f"{'Model':<15} {'BLEU':<8} {'chrF':<8} {'Time(s)':<10} {'Samples':<10}")
            print("-" * 55)
            
            for model_name in ['rnn', 'transformer']:
                if model_name in results:
                    result = results[model_name]
                    if 'error' not in result:
                        print(f"{model_name.upper():<15} {result['bleu']:.2f} {result['chrf']:.2f} {result['time']:.1f} {result.get('sample_count', 0)}")
                    else:
                        print(f"{model_name.upper():<15} {'FAILED':<8} {'FAILED':<8} {'--':<10} {'--':<10}")
            
            # Show improvements if both models successful
            if 'rnn' in successful_models and 'transformer' in successful_models:
                bleu_diff = results['transformer']['bleu'] - results['rnn']['bleu']
                chrf_diff = results['transformer']['chrf'] - results['rnn']['chrf']
                
                print(f"\n🔄 IMPROVEMENT (Transformer vs RNN):")
                print(f"BLEU: {bleu_diff:+.2f}")
                print(f"chrF: {chrf_diff:+.2f}")
        
        else:
            print("❌ Semua model gagal di-evaluate!")
        
        return results, all_references

    def ablation_study(self, model_path, test_loader, vocab_src, vocab_tgt, model_type='transformer'):
        """Perform ablation study"""
        print("\n" + "="*80)
        print("ABLATION STUDY: Model Performance")
        print("="*80)
        
        ablation_results = {}
        
        for beam_size_to_test in [1, 3, 5]:
            print(f"\nEvaluating with Beam Size = {beam_size_to_test}...")
            
            model, src_sp_model, tgt_sp_model = self.load_model_and_vocab(model_path, model_type)
            if model is None:
                print("❌ Model loading failed. Skipping ablation study.")
                return {}
                
            start_time = time.time()
            
            predictions, references = self.evaluate_on_dataset(
                model, test_loader, src_sp_model, tgt_sp_model, beam_size=beam_size_to_test, max_samples=500
            )
            
            bleu_score = self.calculate_bleu(predictions, references)
            chrf_score = self.calculate_chrf(predictions, references)
            eval_time = time.time() - start_time
            
            results_for_bs = {
                'bleu': bleu_score,
                'chrf': chrf_score,
                'time': eval_time,
                'samples_evaluated': len(predictions)
            }
            
            ablation_results[str(beam_size_to_test)] = results_for_bs
            print(f"Beam Size {beam_size_to_test} Results: BLEU: {bleu_score:.2f}, chrF: {chrf_score:.2f}, Time: {eval_time:.1f}s")

        return ablation_results
        
    def save_results(self, results, filename='evaluation_results.json'):
        """Save evaluation results to JSON file"""
        def convert_to_serializable(obj):
            if isinstance(obj, (np.ndarray, torch.Tensor)):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj
        
        results_serializable = convert_to_serializable(results)
        
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        
        with open(filename, 'w', encoding='utf-8') as f:
            json.dump(results_serializable, f, indent=2, ensure_ascii=False)
        
        print(f"Results saved to {filename}")