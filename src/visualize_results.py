import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import json
import numpy as np
from pathlib import Path
import os

# Set style
plt.style.use('default')
sns.set_palette("husl")

class ResultsVisualizer:
    def __init__(self, results_file='outputs/evaluation_results.json'):
        """Initialize visualizer with results file"""
        self.results_file = results_file
        self.results = self.load_results()
        
        # Create output directory
        os.makedirs('outputs/plots', exist_ok=True)
    
    def load_results(self):
        """Load results from JSON file"""
        try:
            with open(self.results_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except FileNotFoundError:
            print(f"Results file {self.results_file} not found!")
            return None
    
    def plot_model_comparison(self):
        """Plot comparison between RNN and Transformer models"""
        if not self.results or 'model_comparison' not in self.results:
            print("Model comparison results not found!")
            return
        
        comparison = self.results['model_comparison']
        
        # Prepare data with error handling for FAILED models
        models = []
        bleu_scores = []
        chrf_scores = []
        times = []
        
        for model_name in ['rnn', 'transformer']:
            result = comparison.get(model_name, {})
            if 'error' not in result and result.get('bleu') is not None:
                models.append(model_name.upper() + '+Attention' if model_name == 'rnn' else model_name.upper())
                bleu_scores.append(result['bleu'])
                chrf_scores.append(result['chrf'])
                times.append(result['time'])
            else:
                # Handle failed models to prevent plotting errors
                models.append(model_name.upper() + ' (FAILED)')
                bleu_scores.append(0)
                chrf_scores.append(0)
                times.append(0)

        if not models:
            print("No valid model data to plot.")
            return

        # Create subplots
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # BLEU scores
        bars1 = axes[0].bar(models, bleu_scores, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
        axes[0].set_title('BLEU Scores Comparison', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('BLEU Score', fontsize=12)
        axes[0].set_ylim(0, max(bleu_scores) * 1.2 if any(bleu_scores) else 10)
        
        # Add value labels
        for bar, score in zip(bars1, bleu_scores):
            axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                         f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # chrF scores
        bars2 = axes[1].bar(models, chrf_scores, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
        axes[1].set_title('chrF Scores Comparison', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('chrF Score', fontsize=12)
        axes[1].set_ylim(0, max(chrf_scores) * 1.2 if any(chrf_scores) else 10)
        
        # Add value labels
        for bar, score in zip(bars2, chrf_scores):
            axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                         f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
        
        # Inference times
        bars3 = axes[2].bar(models, times, color=['#FF6B6B', '#4ECDC4'], alpha=0.8)
        axes[2].set_title('Inference Time Comparison', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Time (seconds)', fontsize=12)
        axes[2].set_ylim(0, max(times) * 1.2 if any(times) else 10)
        
        # Add value labels
        for bar, time in zip(bars3, times):
            axes[2].text(bar.get_x() + bar.get_width()/2, bar.get_height() + max(times)*0.02,
                         f'{time:.1f}s', ha='center', va='bottom', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('outputs/plots/model_comparison.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        # Calculate and display improvements with zero-division check
        rnn_bleu = comparison['rnn'].get('bleu', 0)
        rnn_chrf = comparison['rnn'].get('chrf', 0)
        
        if rnn_bleu > 0:
            bleu_improvement = ((comparison['transformer']['bleu'] - rnn_bleu) / rnn_bleu) * 100
        else:
            bleu_improvement = float('inf')  # Prevent ZeroDivisionError
            
        if rnn_chrf > 0:
            chrf_improvement = ((comparison['transformer']['chrf'] - rnn_chrf) / rnn_chrf) * 100
        else:
            chrf_improvement = float('inf')  # Prevent ZeroDivisionError
            
        print("Model Comparison Summary:")
        if bleu_improvement != float('inf'):
            print(f"BLEU improvement: {bleu_improvement:+.1f}%")
        else:
            print("BLEU improvement: N/A (RNN BLEU is 0)")
        
        if chrf_improvement != float('inf'):
            print(f"chrF improvement: {chrf_improvement:+.1f}%")
        else:
            print("chrF improvement: N/A (RNN chrF is 0)")
    
    def plot_ablation_study(self):
        """Plot ablation study results"""
        if not self.results or 'ablation_study' not in self.results or not self.results['ablation_study']:
            print("Ablation study results not found or empty!")
            return
        
        ablation = self.results['ablation_study']
        
        beam_sizes = list(ablation.keys())
        beam_sizes = [int(k) for k in beam_sizes]
        beam_sizes.sort()
        
        bleu_scores = [ablation[str(bs)]['bleu'] for bs in beam_sizes]
        chrf_scores = [ablation[str(bs)]['chrf'] for bs in beam_sizes]
        times = [ablation[str(bs)]['time'] for bs in beam_sizes]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].plot(beam_sizes, bleu_scores, 'o-', linewidth=2, markersize=8, color='#FF6B6B')
        axes[0].set_title('BLEU Score vs Beam Size', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Beam Size', fontsize=12)
        axes[0].set_ylabel('BLEU Score', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xticks(beam_sizes)
        
        for x, y in zip(beam_sizes, bleu_scores):
            axes[0].annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
                             xytext=(0,10), ha='center', fontweight='bold')
        
        axes[1].plot(beam_sizes, chrf_scores, 'o-', linewidth=2, markersize=8, color='#4ECDC4')
        axes[1].set_title('chrF Score vs Beam Size', fontsize=14, fontweight='bold')
        axes[1].set_xlabel('Beam Size', fontsize=12)
        axes[1].set_ylabel('chrF Score', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xticks(beam_sizes)
        
        for x, y in zip(beam_sizes, chrf_scores):
            axes[1].annotate(f'{y:.2f}', (x, y), textcoords="offset points", 
                             xytext=(0,10), ha='center', fontweight='bold')
        
        axes[2].plot(beam_sizes, times, 'o-', linewidth=2, markersize=8, color='#FFD93D')
        axes[2].set_title('Inference Time vs Beam Size', fontsize=14, fontweight='bold')
        axes[2].set_xlabel('Beam Size', fontsize=12)
        axes[2].set_ylabel('Time (seconds)', fontsize=12)
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xticks(beam_sizes)
        
        for x, y in zip(beam_sizes, times):
            axes[2].annotate(f'{y:.1f}s', (x, y), textcoords="offset points", 
                             xytext=(0,10), ha='center', fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('outputs/plots/ablation_study.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_results_table(self):
        """Create a summary table of all results"""
        if not self.results:
            print("No results to display!")
            return
        
        if 'model_comparison' in self.results:
            print("\n" + "="*60)
            print("MODEL COMPARISON RESULTS")
            print("="*60)
            
            comparison = self.results['model_comparison']
            
            print(f"{'Model':<20} {'BLEU':<8} {'chrF':<8} {'Time(s)':<10}")
            print("-" * 60)
            
            rnn_data = comparison.get('rnn', {'bleu': 0, 'chrf': 0, 'time': 0, 'error': 'Failed'})
            transformer_data = comparison.get('transformer', {'bleu': 0, 'chrf': 0, 'time': 0, 'error': 'Failed'})

            if 'error' in rnn_data:
                rnn_bleu_str = 'FAILED'
                rnn_chrf_str = 'FAILED'
                rnn_time_str = '--'
            else:
                rnn_bleu_str = f"{rnn_data['bleu']:.2f}"
                rnn_chrf_str = f"{rnn_data['chrf']:.2f}"
                rnn_time_str = f"{rnn_data['time']:.1f}"

            if 'error' in transformer_data:
                transformer_bleu_str = 'FAILED'
                transformer_chrf_str = 'FAILED'
                transformer_time_str = '--'
            else:
                transformer_bleu_str = f"{transformer_data['bleu']:.2f}"
                transformer_chrf_str = f"{transformer_data['chrf']:.2f}"
                transformer_time_str = f"{transformer_data['time']:.1f}"
                
            print(f"{'RNN+Attention':<20} {rnn_bleu_str:<8} {rnn_chrf_str:<8} {rnn_time_str:<10}")
            print(f"{'Transformer':<20} {transformer_bleu_str:<8} {transformer_chrf_str:<8} {transformer_time_str:<10}")
            
            if 'error' not in rnn_data and 'error' not in transformer_data and rnn_data['bleu'] > 0 and rnn_data['chrf'] > 0:
                bleu_improvement = transformer_data['bleu'] - rnn_data['bleu']
                chrf_improvement = transformer_data['chrf'] - rnn_data['chrf']
                print(f"\nTransformer Improvements:")
                print(f"BLEU: +{bleu_improvement:.2f} ({(bleu_improvement/rnn_data['bleu']*100):+.1f}%)")
                print(f"chrF: +{chrf_improvement:.2f} ({(chrf_improvement/rnn_data['chrf']*100):+.1f}%)")

        if 'ablation_study' in self.results:
            print("\n" + "="*60)
            print("ABLATION STUDY RESULTS (Transformer)")
            print("="*60)
            
            ablation = self.results['ablation_study']
            
            if ablation:
                print(f"{'Beam Size':<12} {'BLEU':<8} {'chrF':<8} {'Time(s)':<10}")
                print("-" * 60)
                
                beam_sizes = sorted([int(k) for k in ablation.keys()])
                for bs in beam_sizes:
                    data = ablation[str(bs)]
                    print(f"{bs:<12} {data['bleu']:<8.2f} {data['chrf']:<8.2f} {data['time']:<10.1f}")
            else:
                print("Ablation study data not available.")
    
    def plot_training_curves(self, training_log_file='runs/training_log.json'):
        """Plot training curves if available"""
        try:
            with open(training_log_file, 'r') as f:
                training_data = json.load(f)
        except FileNotFoundError:
            print(f"Training log file {training_log_file} not found!")
            return
        
        epochs = training_data.get('epochs', [])
        train_losses = training_data.get('train_losses', [])
        val_losses = training_data.get('val_losses', [])
        train_ppls = training_data.get('train_ppls', [])
        val_ppls = training_data.get('val_ppls', [])
        
        if not epochs:
            print("No training data available for plotting!")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        axes[0].plot(epochs, train_losses, 'o-', label='Training Loss', linewidth=2, color='#FF6B6B')
        axes[0].plot(epochs, val_losses, 'o-', label='Validation Loss', linewidth=2, color='#4ECDC4')
        axes[0].set_title('Training and Validation Loss', fontsize=14, fontweight='bold')
        axes[0].set_xlabel('Epoch', fontsize=12)
        axes[0].set_ylabel('Loss', fontsize=12)
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        if train_ppls and val_ppls:
            axes[1].plot(epochs, train_ppls, 'o-', label='Training PPL', linewidth=2, color='#FF6B6B')
            axes[1].plot(epochs, val_ppls, 'o-', label='Validation PPL', linewidth=2, color='#4ECDC4')
            axes[1].set_title('Training and Validation Perplexity', fontsize=14, fontweight='bold')
            axes[1].set_xlabel('Epoch', fontsize=12)
            axes[1].set_ylabel('Perplexity', fontsize=12)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        else:
            axes[1].text(0.5, 0.5, 'Perplexity data\nnot available', 
                         ha='center', va='center', transform=axes[1].transAxes,
                         fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            axes[1].set_title('Perplexity (Not Available)', fontsize=14)
        
        plt.tight_layout()
        plt.savefig('outputs/plots/training_curves.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def create_comprehensive_report(self):
        """Create a comprehensive visual report"""
        print("Creating comprehensive evaluation report...")
        
        print("1. Creating model comparison plot...")
        self.plot_model_comparison()
        
        print("2. Creating ablation study plot...")
        self.plot_ablation_study()
        
        print("3. Creating training curves plot...")
        self.plot_training_curves()
        
        print("4. Creating summary tables...")
        self.create_results_table()
        
        print("5. Generating summary statistics...")
        self.generate_summary_stats()
        
        print("\nVisualization complete! Check outputs/plots/ folder for generated plots.")
    
    def generate_summary_stats(self):
        """Generate summary statistics"""
        if not self.results:
            return
        
        print("\n" + "="*80)
        print("EVALUATION SUMMARY")
        print("="*80)
        
        if 'model_comparison' in self.results:
            comparison = self.results['model_comparison']
            
            rnn_data = comparison.get('rnn', {})
            transformer_data = comparison.get('transformer', {})

            if 'error' not in rnn_data and 'error' not in transformer_data:
                best_model = 'Transformer' if transformer_data['bleu'] > rnn_data['bleu'] else 'RNN+Attention'
                print(f"Best performing model: {best_model}")
                
                if best_model == 'Transformer':
                    bleu_gain = transformer_data['bleu'] - rnn_data['bleu']
                    chrf_gain = transformer_data['chrf'] - rnn_data['chrf']
                    print("Performance gains over baseline:")
                    print(f"  - BLEU: +{bleu_gain:.2f} points")
                    print(f"  - chrF: +{chrf_gain:.2f} points")
            else:
                print("Cannot determine best model due to evaluation failure.")
        
        if 'ablation_study' in self.results and self.results['ablation_study']:
            ablation = self.results['ablation_study']
            
            best_beam_size = max(ablation.keys(), key=lambda x: ablation[x]['bleu'])
            best_bleu = ablation[best_beam_size]['bleu']
            
            print(f"\nOptimal beam size: {best_beam_size}")
            print(f"Best BLEU score: {best_bleu:.2f}")
            
            if '1' in ablation:
                beam_1_time = ablation['1']['time']
                best_beam_time = ablation[best_beam_size]['time']
                if beam_1_time > 0:
                    speedup = best_beam_time / beam_1_time
                    print(f"Speed trade-off: {speedup:.1f}x slower than greedy decoding")
                else:
                    print("Speed trade-off: N/A (Greedy decoding time is 0)")
        
        print("\n" + "="*80)

def main():
    """Main visualization function"""
    results_file = 'outputs/evaluation_results.json'
    
    if not os.path.exists(results_file):
        print(f"Results file {results_file} not found!")
        print("Please run evaluation.py first to generate results.")
        return
    
    visualizer = ResultsVisualizer(results_file)
    visualizer.create_comprehensive_report()


if __name__ == "__main__":
    main()