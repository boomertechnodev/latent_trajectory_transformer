#!/usr/bin/env python3
"""
Comprehensive test suite for neural code search.

Tests both semantic understanding and explanation quality across
different query types.
"""

import subprocess
import sys
from typing import List, Dict, Tuple

# ===================================================================
# TEST QUERIES - Designed to test different aspects
# ===================================================================

TEST_QUERIES = [
    # SEMANTIC QUERIES - Should work via neural understanding, NOT keyword match
    {
        'query': 'how do I evolve latent states over time?',
        'expected_files': ['latent_drift_trajectory.py'],
        'expected_concepts': ['SDE', 'ODE', 'dynamics', 'solve', 'trajectory'],
        'type': 'semantic',
        'description': 'Semantic query about temporal evolution'
    },
    {
        'query': 'where is stochastic differential equation implemented?',
        'expected_files': ['latent_drift_trajectory.py', 'raccoon_alternative.py'],
        'expected_concepts': ['RaccoonDynamics', 'drift', 'diffusion', 'solve_sde'],
        'type': 'semantic',
        'description': 'Semantic query about SDE (uses different terminology than code)'
    },
    {
        'query': 'code for preventing catastrophic forgetting',
        'expected_files': ['latent_drift_trajectory.py', 'raccoon_alternative.py'],
        'expected_concepts': ['memory', 'replay', 'experience', 'RaccoonMemory'],
        'type': 'semantic',
        'description': 'Conceptual query (term not in code)'
    },
    {
        'query': 'invertible transformations for density estimation',
        'expected_files': ['latent_drift_trajectory.py'],
        'expected_concepts': ['flow', 'coupling', 'bijective', 'log_det'],
        'type': 'semantic',
        'description': 'Mathematical concept query'
    },
    {
        'query': 'how to reduce attention complexity from quadratic to logarithmic',
        'expected_files': ['fractal_attention2.py', 'FRACTAL_ATTENTION_TUTORIAL.md'],
        'expected_concepts': ['Hilbert', 'fractal', 'O(n^2)', 'O(log n)'],
        'type': 'semantic',
        'description': 'Complexity reduction concept'
    },

    # EXACT MATCH QUERIES - Should work via keyword matching
    {
        'query': 'RaccoonDynamics class definition',
        'expected_files': ['latent_drift_trajectory.py'],
        'expected_concepts': ['class RaccoonDynamics', 'drift', 'diffusion'],
        'type': 'exact',
        'description': 'Direct class name query'
    },
    {
        'query': 'FastEppsPulley normality test',
        'expected_files': ['latent_drift_trajectory.py'],
        'expected_concepts': ['FastEppsPulley', 'characteristic function', 'weight'],
        'type': 'exact',
        'description': 'Exact method name'
    },

    # MIXED QUERIES - Combine semantic and exact
    {
        'query': 'Hilbert curve space filling pattern for attention',
        'expected_files': ['fractal_attention2.py'],
        'expected_concepts': ['hilbert', 'space-filling', 'attention', 'fractal'],
        'type': 'mixed',
        'description': 'Mix of mathematical concept + specific technique'
    },
]


def run_neural_query(query: str, index_path: str, top_k: int = 3) -> List[Dict]:
    """Run neural search and parse results."""
    try:
        result = subprocess.run(
            ['python', 'neural_code_search.py', 'query', query,
             '--index', index_path, '--top-k', str(top_k)],
            capture_output=True,
            text=True,
            timeout=30
        )

        if result.returncode != 0:
            print(f"ERROR: {result.stderr}")
            return []

        # Parse results
        results = []
        lines = result.stdout.split('\n')
        current_result = {}

        for line in lines:
            if line.strip().startswith('File:'):
                if current_result:
                    results.append(current_result)
                current_result = {'filepath': line.split('File:')[1].strip()}
            elif line.strip().startswith('Relevance:'):
                current_result['relevance'] = float(line.split('Relevance:')[1].strip())
            elif line.strip().startswith('Explanation:'):
                # Next lines are explanation
                current_result['explanation'] = ''
            elif 'explanation' in current_result and current_result.get('explanation') is not None:
                if line.strip() and not line.startswith('-'):
                    current_result['explanation'] += line.strip() + ' '

        if current_result:
            results.append(current_result)

        return results

    except Exception as e:
        print(f"ERROR running query: {e}")
        return []


def evaluate_query(test_case: Dict, results: List[Dict]) -> Dict:
    """Evaluate query results against expectations."""

    eval_result = {
        'query': test_case['query'],
        'type': test_case['type'],
        'description': test_case['description'],
        'passed': False,
        'found_expected_files': [],
        'found_concepts': [],
        'top_relevance': 0.0,
        'explanation_quality': 'unknown',
        'results_count': len(results)
    }

    if not results:
        return eval_result

    # Check if expected files are in top results
    result_files = [r['filepath'].lstrip('./') for r in results]
    for expected_file in test_case['expected_files']:
        if any(expected_file in rf for rf in result_files):
            eval_result['found_expected_files'].append(expected_file)

    # Check for expected concepts in any result
    all_text = ' '.join([
        r.get('filepath', '') + ' ' + r.get('explanation', '')
        for r in results
    ]).lower()

    for concept in test_case['expected_concepts']:
        if concept.lower() in all_text:
            eval_result['found_concepts'].append(concept)

    # Get top relevance score
    if results:
        eval_result['top_relevance'] = max(r.get('relevance', 0.0) for r in results)

    # Evaluate explanation quality (basic heuristic)
    if results and 'explanation' in results[0]:
        explanation = results[0].get('explanation', '')
        if len(explanation) > 20 and not explanation.startswith('ision rrpy'):
            # Not gibberish
            eval_result['explanation_quality'] = 'coherent'
        else:
            eval_result['explanation_quality'] = 'gibberish'

    # Determine if passed
    file_match = len(eval_result['found_expected_files']) > 0
    concept_match = len(eval_result['found_concepts']) >= len(test_case['expected_concepts']) // 2

    eval_result['passed'] = file_match and concept_match

    return eval_result


def main():
    index_path = sys.argv[1] if len(sys.argv) > 1 else 'neural_intensive.index'

    print("=" * 70)
    print("NEURAL CODE SEARCH - COMPREHENSIVE TEST SUITE")
    print("=" * 70)
    print(f"Index: {index_path}")
    print(f"Total queries: {len(TEST_QUERIES)}")
    print()

    results_by_type = {'semantic': [], 'exact': [], 'mixed': []}

    for i, test_case in enumerate(TEST_QUERIES, 1):
        print(f"\n[Query {i}/{len(TEST_QUERIES)}] {test_case['type'].upper()}")
        print(f"Question: \"{test_case['query']}\"")
        print(f"Description: {test_case['description']}")
        print()

        # Run query
        results = run_neural_query(test_case['query'], index_path, top_k=3)

        # Evaluate
        eval_result = evaluate_query(test_case, results)
        results_by_type[test_case['type']].append(eval_result)

        # Display
        if eval_result['passed']:
            print("✅ PASSED")
        else:
            print("❌ FAILED")

        print(f"   Found files: {eval_result['found_expected_files']}")
        print(f"   Found concepts: {eval_result['found_concepts']}")
        print(f"   Top relevance: {eval_result['top_relevance']:.3f}")
        print(f"   Explanation: {eval_result['explanation_quality']}")

        if results:
            print(f"\n   Top result: {results[0]['filepath']}")
            if 'explanation' in results[0]:
                expl = results[0]['explanation'][:100]
                print(f"   Explanation sample: {expl}...")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for query_type in ['semantic', 'exact', 'mixed']:
        type_results = results_by_type[query_type]
        if not type_results:
            continue

        passed = sum(1 for r in type_results if r['passed'])
        total = len(type_results)
        pct = (passed / total) * 100 if total > 0 else 0

        print(f"\n{query_type.upper()} Queries:")
        print(f"  Passed: {passed}/{total} ({pct:.0f}%)")

        coherent = sum(1 for r in type_results if r['explanation_quality'] == 'coherent')
        if total > 0:
            print(f"  Coherent explanations: {coherent}/{total} ({coherent/total*100:.0f}%)")

    # Overall
    all_results = sum(results_by_type.values(), [])
    total_passed = sum(1 for r in all_results if r['passed'])
    total_queries = len(all_results)

    print(f"\nOVERALL: {total_passed}/{total_queries} passed ({total_passed/total_queries*100:.0f}%)")

    coherent_total = sum(1 for r in all_results if r['explanation_quality'] == 'coherent')
    print(f"Coherent explanations: {coherent_total}/{total_queries} ({coherent_total/total_queries*100:.0f}%)")


if __name__ == '__main__':
    main()
