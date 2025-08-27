#!/usr/bin/env python3
"""
Script untuk menjalankan evaluasi lengkap model machine translation
Langkah 4: Evaluasi & Analisis

Usage:
    python run_evaluation.py
"""

import os
import sys
import torch
from torch.utils.data import DataLoader
import argparse
from pathlib import Path
import sentencepiece as spm

# Add src directory to path
sys.path.append(str(Path(__file__).resolve().parent / 'src'))

# Import modules (sesuaikan dengan struktur project Anda)
try:
    from evaluation import ModelEvaluator
    from visualize_results import ResultsVisualizer
    from data_loader import TranslationDataset, get_data_loaders
    from utils import save_checkpoint, load_checkpoint
except ImportError as e:
    print(f"Import error: {e}")
    print("Please make sure all required modules are available.")
    print("You may need to adjust the import statements based on your project structure.")
    sys.exit(1)

def check_requirements():
    """Check if all required files and directories exist"""
    required_files = [
        'models/best_model.pth',  # RNN model
        'models/best_transformer_model.pth',  # Transformer model
        'data/processed/test.tsv',  # Test data
        'data/tokenizer/eng_sp.model',  # Source tokenizer
        'data/tokenizer/ind_sp.model',  # Target tokenizer
    ]
    
    required_dirs = [
        'outputs',
        'outputs/plots'
    ]
    
    # Check directories
    for dir_path in required_dirs:
        os.makedirs(dir_path, exist_ok=True)
    
    # Check files
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("Missing required files:")
        for file_path in missing_files:
            print(f"  - {file_path}")
        return False
    
    return True

def load_test_data(batch_size=32):
    """Load test dataset"""
    print("Loading test dataset...")
    
    try:
        # Load tokenizer models
        src_sp_model = spm.SentencePieceProcessor()
        src_sp_model.load('data/tokenizer/eng_sp.model')
        
        tgt_sp_model = spm.SentencePieceProcessor()
        tgt_sp_model.load('data/tokenizer/ind_sp.model')
        
        # Create test dataset
        test_dataset = TranslationDataset(
            file_path='data/processed/test.tsv',
            src_sp_model=src_sp_model,
            tgt_sp_model=tgt_sp_model
        )
        
        # Define collate function
        def collate_fn(batch):
            src_ids = [item['src_ids'] for item in batch]
            tgt_ids = [item['tgt_ids'] for item in batch]
            src_texts = [item['src_text'] for item in batch]
            tgt_texts = [item['tgt_text'] for item in batch]
            
            # Pad sequences
            src_padded = torch.nn.utils.rnn.pad_sequence(src_ids, batch_first=True, padding_value=src_sp_model.pad_id())
            tgt_padded = torch.nn.utils.rnn.pad_sequence(tgt_ids, batch_first=True, padding_value=tgt_sp_model.pad_id())
            
            return {
                'src_ids': src_padded,
                'tgt_ids': tgt_padded,
                'src_texts': src_texts,
                'tgt_texts': tgt_texts
            }
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn
        )
        
        print(f"Test dataset loaded: {len(test_dataset)} samples")
        return test_loader, src_sp_model, tgt_sp_model
        
    except Exception as e:
        print(f"Error loading test data: {e}")
        return None, None, None

def load_vocabularies():
    """Load source and target vocabularies (SentencePiece models)"""
    print("Loading vocabularies...")
    
    try:
        # Load tokenizer models
        src_sp_model = spm.SentencePieceProcessor()
        src_sp_model.load('data/tokenizer/eng_sp.model')
        
        tgt_sp_model = spm.SentencePieceProcessor()
        tgt_sp_model.load('data/tokenizer/ind_sp.model')
        
        print(f"Vocabularies loaded: src={src_sp_model.get_piece_size()}, tgt={tgt_sp_model.get_piece_size()}")
        return src_sp_model, tgt_sp_model
        
    except Exception as e:
        print(f"Error loading vocabularies: {e}")
        return None, None

def run_full_evaluation(args):
    """Run complete evaluation pipeline with robust error handling"""
    print("Starting full evaluation pipeline...")
    print("="*80)
    
    # Initialize evaluator
    evaluator = ModelEvaluator(device=args.device)
    
    # Load test data
    test_loader, src_sp_model, tgt_sp_model = load_test_data(args.batch_size)
    if test_loader is None:
        print("Failed to load test data!")
        return False
    
    vocab_src, vocab_tgt = src_sp_model, tgt_sp_model
    
    print("\n1. MODEL COMPARISON")
    print("-" * 40)
    
    # Compare models with error handling
    try:
        comparison_results, references = evaluator.compare_models(
            rnn_model_path='models/best_model.pth',
            transformer_model_path='models/best_transformer_model.pth',
            test_loader=test_loader,
            vocab_src=vocab_src,
            vocab_tgt=vocab_tgt,
            beam_size=args.beam_size
        )
        
        # Check if we have any successful results
        successful_results = {}
        for model_name, result in comparison_results.items():
            if 'error' not in result and result.get('predictions'):
                successful_results[model_name] = result
        
        if not successful_results:
            print("Semua model gagal di-evaluate!")
            return False
            
    except Exception as e:
        print(f"Error in model comparison: {e}")
        return False
    
    # Skip ablation study if transformer fails
    transformer_success = 'transformer' in successful_results
    
    print("\n2. ABLATION STUDY")
    print("-" * 40)
    
    if transformer_success:
        try:
            ablation_results = evaluator.ablation_study(
                model_path='models/best_transformer_model.pth',
                test_loader=test_loader,
                vocab_src=vocab_src,
                vocab_tgt=vocab_tgt,
                model_type='transformer'
            )
        except Exception as e:
            print(f"Ablation study failed: {e}")
            ablation_results = {}
    else:
        print("Skipping ablation study - transformer model failed")
        ablation_results = {}
    
    print("\n3. TRANSLATION EXAMPLES")
    print("-" * 40)
    
    # Show examples from successful models
    for model_name, result in successful_results.items():
        if result.get('predictions'):
            print(f"\n=== {model_name.upper()} TRANSLATION EXAMPLES ===")
            evaluator.analyze_translations(
                result['predictions'],
                references,
                num_examples=min(args.num_examples, 5)
            )
    
    # Save results
    print("\n4. SAVING RESULTS")
    print("-" * 40)
    
    all_results = {
        'model_comparison': comparison_results,
        'ablation_study': ablation_results,
        'evaluation_settings': {
            'beam_size': args.beam_size,
            'batch_size': args.batch_size,
            'device': args.device,
            'num_examples': args.num_examples
        },
        'successful_models': list(successful_results.keys()),
        'evaluation_date': __import__('time').strftime('%Y-%m-%d %H:%M:%S')
    }
    
    try:
        evaluator.save_results(all_results, 'outputs/evaluation_results.json')
        print("Results saved successfully!")
    except Exception as e:
        print(f"Failed to save results: {e}")
    
    return True

def run_visualization():
    """Run visualization of results"""
    print("\n5. CREATING VISUALIZATIONS")
    print("-" * 40)
    
    try:
        visualizer = ResultsVisualizer('outputs/evaluation_results.json')
        visualizer.create_comprehensive_report()
        return True
    except Exception as e:
        print(f"Error in visualization: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Run complete evaluation pipeline')
    parser.add_argument('--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu',
                        help='Device to use for evaluation (cuda/cpu)')
    parser.add_argument('--batch-size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--beam-size', type=int, default=5,
                        help='Beam size for decoding')
    parser.add_argument('--num-examples', type=int, default=10,
                        help='Number of translation examples to show')
    parser.add_argument('--skip-visualization', action='store_true',
                        help='Skip visualization step')
    
    args = parser.parse_args()
    
    print("Machine Translation Model Evaluation")
    print("="*80)
    print(f"Device: {args.device}")
    print(f"Batch size: {args.batch_size}")
    print(f"Beam size: {args.beam_size}")
    print("="*80)
    
    # Check requirements
    if not check_requirements():
        print("Please ensure all required files are present before running evaluation.")
        sys.exit(1)
    
    # Run evaluation
    success = run_full_evaluation(args)
    
    if not success:
        print("Evaluation failed!")
        sys.exit(1)
    
    # Run visualization (unless skipped)
    if not args.skip_visualization:
        viz_success = run_visualization()
        if not viz_success:
            print("Visualization failed, but evaluation results are saved.")
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETED SUCCESSFULLY!")
    print("="*80)
    print("Results saved to:")
    print("  - outputs/evaluation_results.json")
    print("  - outputs/plots/")
    print("\nYEYYYY BERHASILLL.")
    print("UDAH KESIMPEN HASILNYAAAA DI outputs")

if __name__ == "__main__":
    main()