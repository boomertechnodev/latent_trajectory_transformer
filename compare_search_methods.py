#!/usr/bin/env python3
"""
Compare neural search vs grep baseline.

SLOPPY WAY: Just assume neural is better without testing
NON-SLOPPY WAY: Actually measure precision, recall, and show examples
"""

import subprocess
import sys
from typing import List, Tuple, Dict

# Test queries with ground truth files that SHOULD match
TEST_QUERIES = [
    {
        'query': 'SDE dynamics drift diffusion',
        'ground_truth': ['latent_drift_trajectory.py'],  # Where RaccoonDynamics actually is
        'description': 'Finding stochastic differential equation implementation'
    },
    {
        'query': 'normalizing flow coupling layers',
        'ground_truth': ['latent_drift_trajectory.py', 'raccoon_alternative.py'],
        'description': 'Finding normalizing flow implementations'
    },
    {
        'query': 'experience replay memory buffer',
        'ground_truth': ['latent_drift_trajectory.py', 'raccoon_alternative.py'],
        'description': 'Finding memory buffer for continual learning'
    },
    {
        'query': 'Hilbert curve attention',
        'ground_truth': ['fractal_attention2.py', 'FRACTAL_ATTENTION_TUTORIAL.md'],
        'description': 'Finding fractal attention mechanisms'
    },
    {
        'query': 'Epps-Pulley statistical test',
        'ground_truth': ['latent_drift_trajectory.py'],
        'description': 'Finding statistical regularization tests'
    },
]


def run_grep_search(query: str, top_k: int = 5) -> List[Tuple[str, int]]:
    """
    Run grep search and return top-k files by match count.

    Returns:
        List of (filepath, match_count) tuples
    """
    try:
        # Use ripgrep if available, otherwise grep
        result = subprocess.run(
            ['grep', '-rl', '-i', query, '.'],
            capture_output=True,
            text=True,
            timeout=10
        )

        if result.returncode != 0:
            return []

        files = result.stdout.strip().split('\n')
        files = [f for f in files if f and not f.startswith('./.git')]

        # Count matches per file
        file_counts = []
        for filepath in files[:50]:  # Limit to avoid timeout
            try:
                count_result = subprocess.run(
                    ['grep', '-c', '-i', query, filepath],
                    capture_output=True,
                    text=True,
                    timeout=1
                )
                count = int(count_result.stdout.strip()) if count_result.returncode == 0 else 0
                file_counts.append((filepath, count))
            except Exception:
                continue

        # Sort by match count
        file_counts.sort(key=lambda x: x[1], reverse=True)
        return file_counts[:top_k]

    except Exception as e:
        print(f"  grep error: {e}")
        return []


def run_neural_search(query: str, index_path: str, top_k: int = 5) -> List[Tuple[str, float]]:
    """
    Run neural search and return top-k results.

    Returns:
        List of (filepath, relevance_score) tuples
    """
    try:
        result = subprocess.run(
            ['python', 'neural_code_search.py', 'query', query,
             '--index', index_path, '--top-k', str(top_k), '--no-explain'],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            print(f"  neural error: {result.stderr}")
            return []

        # Parse output
        results = []
        lines = result.stdout.split('\n')
        current_file = None
        current_score = None

        for line in lines:
            if line.strip().startswith('File:'):
                current_file = line.split('File:')[1].strip()
            elif line.strip().startswith('Relevance:'):
                current_score = float(line.split('Relevance:')[1].strip())
                if current_file and current_score:
                    results.append((current_file, current_score))
                    current_file = None
                    current_score = None

        return results

    except Exception as e:
        print(f"  neural error: {e}")
        return []


def calculate_precision_at_k(results: List[str], ground_truth: List[str], k: int = 5) -> float:
    """
    Calculate precision@k: how many of top-k results are relevant?
    """
    if not results:
        return 0.0

    # Normalize file paths
    results_normalized = [r.lstrip('./') for r in results[:k]]
    truth_normalized = [t.lstrip('./') for t in ground_truth]

    hits = sum(1 for r in results_normalized if any(t in r for t in truth_normalized))
    return hits / min(k, len(results))


def main():
    print("=" * 70)
    print("SEARCH METHOD COMPARISON: grep vs Neural Search")
    print("=" * 70)
    print()

    index_path = 'neural_code_retrained.index'

    # Check if neural index exists
    import os
    if not os.path.exists(index_path):
        print(f"⚠️  Neural index not found: {index_path}")
        print("   Run: python neural_code_search.py index . --output neural_code_retrained.index")
        return

    results_summary = []

    for test_case in TEST_QUERIES:
        query = test_case['query']
        ground_truth = test_case['ground_truth']
        description = test_case['description']

        print(f"\n{'='*70}")
        print(f"Query: \"{query}\"")
        print(f"Task: {description}")
        print(f"Ground truth: {', '.join(ground_truth)}")
        print(f"{'='*70}\n")

        # Run grep
        print("Running grep...")
        grep_results = run_grep_search(query, top_k=5)
        grep_files = [r[0] for r in grep_results]
        grep_precision = calculate_precision_at_k(grep_files, ground_truth, k=5)

        print(f"grep results (top 5):")
        for filepath, count in grep_results:
            in_truth = "✓" if any(t in filepath for t in ground_truth) else "✗"
            print(f"  {in_truth} {filepath} ({count} matches)")
        print(f"  Precision@5: {grep_precision:.2%}")

        # Run neural search
        print("\nRunning neural search...")
        neural_results = run_neural_search(query, index_path, top_k=5)
        neural_files = [r[0] for r in neural_results]
        neural_precision = calculate_precision_at_k(neural_files, ground_truth, k=5)

        print(f"neural results (top 5):")
        for filepath, score in neural_results:
            in_truth = "✓" if any(t in filepath for t in ground_truth) else "✗"
            print(f"  {in_truth} {filepath} (score: {score:.3f})")
        print(f"  Precision@5: {neural_precision:.2%}")

        # Summary
        winner = "NEURAL" if neural_precision > grep_precision else ("grep" if grep_precision > neural_precision else "TIE")
        results_summary.append({
            'query': query,
            'grep_p5': grep_precision,
            'neural_p5': neural_precision,
            'winner': winner
        })

        print(f"\n>>> WINNER: {winner} (neural: {neural_precision:.2%} vs grep: {grep_precision:.2%})")

    # Overall summary
    print(f"\n\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}\n")

    avg_grep = sum(r['grep_p5'] for r in results_summary) / len(results_summary)
    avg_neural = sum(r['neural_p5'] for r in results_summary) / len(results_summary)

    neural_wins = sum(1 for r in results_summary if r['winner'] == 'NEURAL')
    grep_wins = sum(1 for r in results_summary if r['winner'] == 'grep')
    ties = sum(1 for r in results_summary if r['winner'] == 'TIE')

    print(f"Average Precision@5:")
    print(f"  grep:   {avg_grep:.2%}")
    print(f"  neural: {avg_neural:.2%}")
    print()
    print(f"Win/Loss/Tie:")
    print(f"  NEURAL wins: {neural_wins}/{len(results_summary)}")
    print(f"  grep wins:   {grep_wins}/{len(results_summary)}")
    print(f"  Ties:        {ties}/{len(results_summary)}")
    print()

    if avg_neural > avg_grep:
        improvement = ((avg_neural - avg_grep) / avg_grep) * 100
        print(f"🎉 Neural search is {improvement:.1f}% better than grep!")
    elif avg_grep > avg_neural:
        decline = ((avg_grep - avg_neural) / avg_grep) * 100
        print(f"⚠️  grep is {decline:.1f}% better than neural search")
        print(f"   (Model needs more training or better architecture)")
    else:
        print("🤷 Neural search and grep perform equally")


if __name__ == '__main__':
    main()
