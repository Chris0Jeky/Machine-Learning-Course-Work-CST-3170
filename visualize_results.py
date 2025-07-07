#!/usr/bin/env python3
"""
Visualization script for Machine Learning Classifier results.
Generates performance charts from experiment results.
"""

import re
import sys
import os
from datetime import datetime

def parse_results_file(filename):
    """Parse results file and extract classifier performance data."""
    classifiers = []
    
    try:
        with open(filename, 'r') as f:
            content = f.read()
            
        # Find the final summary section
        summary_match = re.search(r'Classifier Rankings by Accuracy:(.*?)={10,}', content, re.DOTALL)
        if not summary_match:
            print("Could not find summary section in results file.")
            return []
            
        summary_text = summary_match.group(1)
        
        # Parse each classifier line
        lines = summary_text.strip().split('\n')
        for line in lines:
            match = re.match(r'\s*(\d+)\.\s+(.+?)\s+(\d+\.\d+)%', line)
            if match:
                rank = int(match.group(1))
                name = match.group(2).strip()
                accuracy = float(match.group(3))
                classifiers.append({
                    'rank': rank,
                    'name': name,
                    'accuracy': accuracy
                })
                
    except FileNotFoundError:
        print(f"Error: Could not find file {filename}")
    except Exception as e:
        print(f"Error parsing file: {e}")
        
    return classifiers

def create_matplotlib_visualization(classifiers, output_dir='results'):
    """Create visualization using matplotlib if available."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        # Sort by accuracy
        classifiers.sort(key=lambda x: x['accuracy'], reverse=True)
        
        # Extract data
        names = [c['name'] for c in classifiers]
        accuracies = [c['accuracy'] for c in classifiers]
        
        # Create figure with larger size
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create color gradient
        colors = plt.cm.viridis(np.linspace(0, 1, len(names)))
        
        # Create horizontal bar chart
        bars = ax.barh(names, accuracies, color=colors)
        
        # Add value labels on bars
        for i, (bar, acc) in enumerate(zip(bars, accuracies)):
            ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2, 
                   f'{acc:.2f}%', va='center', fontsize=10)
        
        # Customize the plot
        ax.set_xlabel('Accuracy (%)', fontsize=12)
        ax.set_title('Machine Learning Classifier Performance Comparison', fontsize=16, fontweight='bold')
        ax.set_xlim(0, 105)  # Leave room for labels
        
        # Add grid
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        
        # Invert y-axis to have best performer on top
        ax.invert_yaxis()
        
        # Tight layout
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_file = os.path.join(output_dir, f'classifier_comparison_{timestamp}.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        print(f"Visualization saved to: {output_file}")
        
        # Also create a simple accuracy comparison plot
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        
        # Line plot with markers
        x_pos = np.arange(len(names))
        ax2.plot(x_pos, accuracies, 'o-', linewidth=2, markersize=8, color='darkblue')
        
        # Fill area under the line
        ax2.fill_between(x_pos, 0, accuracies, alpha=0.3, color='lightblue')
        
        # Customize
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels(names, rotation=45, ha='right')
        ax2.set_ylabel('Accuracy (%)', fontsize=12)
        ax2.set_title('Classifier Accuracy Trends', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3, linestyle='--')
        ax2.set_ylim(0, 105)
        
        # Add horizontal line for average
        avg_accuracy = np.mean(accuracies)
        ax2.axhline(y=avg_accuracy, color='red', linestyle='--', alpha=0.7, 
                   label=f'Average: {avg_accuracy:.2f}%')
        ax2.legend()
        
        plt.tight_layout()
        
        # Save the second plot
        output_file2 = os.path.join(output_dir, f'accuracy_trends_{timestamp}.png')
        plt.savefig(output_file2, dpi=300, bbox_inches='tight')
        print(f"Trend visualization saved to: {output_file2}")
        
        return True
        
    except ImportError:
        print("Matplotlib not installed. Falling back to text visualization.")
        return False

def create_text_visualization(classifiers):
    """Create a simple text-based visualization."""
    if not classifiers:
        print("No classifier data to visualize.")
        return
        
    print("\n" + "="*60)
    print("CLASSIFIER PERFORMANCE VISUALIZATION")
    print("="*60 + "\n")
    
    # Find max name length for formatting
    max_name_len = max(len(c['name']) for c in classifiers)
    
    # Sort by accuracy
    classifiers.sort(key=lambda x: x['accuracy'], reverse=True)
    
    # Display bar chart
    for c in classifiers:
        name = c['name'].ljust(max_name_len)
        accuracy = c['accuracy']
        bar_length = int(accuracy / 2)  # Scale to fit in terminal
        bar = "█" * bar_length
        print(f"{name} |{bar} {accuracy:.2f}%")
    
    print("\n" + "-"*60)
    
    # Statistics
    accuracies = [c['accuracy'] for c in classifiers]
    avg_accuracy = sum(accuracies) / len(accuracies)
    max_accuracy = max(accuracies)
    min_accuracy = min(accuracies)
    
    print(f"Average Accuracy: {avg_accuracy:.2f}%")
    print(f"Best Performance: {max_accuracy:.2f}%")
    print(f"Worst Performance: {min_accuracy:.2f}%")
    print(f"Performance Range: {max_accuracy - min_accuracy:.2f}%")
    print("="*60)

def main():
    """Main function to run visualization."""
    # Check if results directory exists
    if not os.path.exists('results'):
        print("Error: 'results' directory not found.")
        print("Please run the experiments first using run_experiments.sh or run_experiments.bat")
        return
    
    # Find the most recent results file
    result_files = [f for f in os.listdir('results') if f.startswith('experiment_results_') and f.endswith('.txt')]
    
    if not result_files:
        print("No results files found in 'results' directory.")
        print("Please run the experiments first.")
        return
    
    # Sort by timestamp and get the most recent
    result_files.sort()
    latest_file = os.path.join('results', result_files[-1])
    
    print(f"Using results file: {latest_file}")
    
    # Parse results
    classifiers = parse_results_file(latest_file)
    
    if not classifiers:
        print("No classifier data found in results file.")
        return
    
    # Try matplotlib visualization first
    if not create_matplotlib_visualization(classifiers):
        # Fall back to text visualization
        create_text_visualization(classifiers)

if __name__ == "__main__":
    main()