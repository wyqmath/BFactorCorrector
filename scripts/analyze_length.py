# analyze_length.py
import json
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

def analyze_sequence_lengths(jsonl_file):
    """Reads a JSONL file and analyzes the sequence lengths."""
    print(f"Analyzing sequence lengths in {jsonl_file}...")
    lengths = []
    try:
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc="Reading data"):
                try:
                    lengths.append(len(json.loads(line)['Sequence']))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping a malformed line in {jsonl_file}")
    except FileNotFoundError:
        print(f"Error: The file {jsonl_file} was not found.")
        return

    if not lengths:
        print("Error: No sequences found in the file.")
        return
        
    lengths = np.array(lengths)
    
    print("\n--- Sequence Length Analysis ---")
    print(f"Total sequences: {len(lengths)}")
    print(f"Min length: {np.min(lengths)}")
    print(f"Max length: {np.max(lengths)}")
    print(f"Mean length: {np.mean(lengths):.2f}")
    print(f"Median length: {np.median(lengths)}")
    
    # Calculate percentiles
    percentiles = [50, 75, 90, 95, 98, 99, 99.5]
    p_values = np.percentile(lengths, percentiles)
    
    print("\nPercentiles:")
    for p, v in zip(percentiles, p_values):
        print(f"{p}th percentile: {int(v)}")

    # --- NEW: Calculate the percentage of sequences with length <= 1024 ---
    count_le_1024 = np.sum(lengths <= 1024)
    total_count = len(lengths)
    percentage_le_1024 = (count_le_1024 / total_count) * 100
    print(f"\nSequences with length <= 1024: {count_le_1024} / {total_count} ({percentage_le_1024:.2f}%)")
        
    # Plotting the histogram
    plt.figure(figsize=(12, 6))
    plt.hist(lengths, bins=100, alpha=0.75, color='skyblue', edgecolor='black')
    plt.title('Distribution of Protein Sequence Lengths')
    plt.xlabel('Sequence Length')
    plt.ylabel('Frequency')
    plt.grid(axis='y', alpha=0.75)
    
    # --- MODIFIED: Removed old percentile lines and added a new one for 1024 ---
    # Add a vertical line for length 1024 and annotate the percentage
    plt.axvline(1024, color='purple', linestyle='--', 
                label=f'Length 1024 (covers {percentage_le_1024:.2f}% of data)')
    
    plt.legend()
    
    # --- MODIFIED: Save the figure with high DPI ---
    plt.savefig('sequence_length_distribution.png', dpi=600)
    print("\nHistogram saved to 'sequence_length_distribution.png' with DPI=600.")
    print("---------------------------------")
    print("\nRecommendation: Choose a value like the 98th or 99th percentile for MAX_LEN.")
    print("For example, based on this analysis, you might choose --max_len 768 or 1024.")

if __name__ == '__main__':
    # Path to your generated training data
    train_file = 'processed_data/train.jsonl'
    analyze_sequence_lengths(train_file)