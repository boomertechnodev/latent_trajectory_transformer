# Research Experiments Skill

Advanced techniques for hypothesis testing, confidence intervals, FDR/Bonferroni corrections, and scientific writing templates for reproducible research.

## Core Competencies

### 1. Hypothesis Testing Framework

#### Parametric Tests
```python
import numpy as np
from scipy import stats
from typing import Dict, Tuple, List, Optional

class ParametricHypothesisTesting:
    """Comprehensive parametric hypothesis testing suite."""

    @staticmethod
    def one_sample_t_test(
        sample: np.ndarray,
        population_mean: float,
        alternative: str = 'two-sided'
    ) -> Dict:
        """One-sample t-test with effect size and confidence intervals."""

        n = len(sample)
        sample_mean = np.mean(sample)
        sample_std = np.std(sample, ddof=1)
        se = sample_std / np.sqrt(n)

        # T-statistic
        t_stat = (sample_mean - population_mean) / se

        # P-value
        df = n - 1
        if alternative == 'two-sided':
            p_value = 2 * stats.t.cdf(-abs(t_stat), df)
        elif alternative == 'greater':
            p_value = 1 - stats.t.cdf(t_stat, df)
        else:  # less
            p_value = stats.t.cdf(t_stat, df)

        # Effect size (Cohen's d)
        cohen_d = (sample_mean - population_mean) / sample_std

        # Confidence interval
        ci_95 = stats.t.interval(0.95, df, loc=sample_mean, scale=se)
        ci_99 = stats.t.interval(0.99, df, loc=sample_mean, scale=se)

        # Power analysis
        from statsmodels.stats.power import ttest_power
        power = ttest_power(cohen_d, n, alpha=0.05, alternative=alternative)

        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'degrees_freedom': df,
            'sample_mean': sample_mean,
            'sample_std': sample_std,
            'cohen_d': cohen_d,
            'ci_95': ci_95,
            'ci_99': ci_99,
            'statistical_power': power,
            'reject_null': p_value < 0.05
        }

    @staticmethod
    def welch_t_test(
        sample1: np.ndarray,
        sample2: np.ndarray,
        alternative: str = 'two-sided'
    ) -> Dict:
        """Welch's t-test for unequal variances with Glass's delta."""

        n1, n2 = len(sample1), len(sample2)
        mean1, mean2 = np.mean(sample1), np.mean(sample2)
        var1, var2 = np.var(sample1, ddof=1), np.var(sample2, ddof=1)

        # Welch's t-statistic
        se = np.sqrt(var1/n1 + var2/n2)
        t_stat = (mean1 - mean2) / se

        # Welch-Satterthwaite degrees of freedom
        df = (var1/n1 + var2/n2)**2 / (
            (var1/n1)**2/(n1-1) + (var2/n2)**2/(n2-1)
        )

        # P-value
        if alternative == 'two-sided':
            p_value = 2 * stats.t.cdf(-abs(t_stat), df)
        elif alternative == 'greater':
            p_value = 1 - stats.t.cdf(t_stat, df)
        else:
            p_value = stats.t.cdf(t_stat, df)

        # Effect sizes
        pooled_std = np.sqrt((var1 + var2) / 2)
        cohen_d = (mean1 - mean2) / pooled_std
        glass_delta = (mean1 - mean2) / np.sqrt(var2)  # Use control group std

        # Confidence interval for difference
        ci_95 = stats.t.interval(0.95, df, loc=mean1-mean2, scale=se)

        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'degrees_freedom': df,
            'mean_difference': mean1 - mean2,
            'cohen_d': cohen_d,
            'glass_delta': glass_delta,
            'ci_95_difference': ci_95,
            'reject_null': p_value < 0.05
        }

    @staticmethod
    def paired_samples_test(
        before: np.ndarray,
        after: np.ndarray,
        alternative: str = 'two-sided'
    ) -> Dict:
        """Paired samples t-test with multiple effect sizes."""

        if len(before) != len(after):
            raise ValueError("Samples must have equal length for paired test")

        differences = after - before
        n = len(differences)
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        se_diff = std_diff / np.sqrt(n)

        # T-statistic
        t_stat = mean_diff / se_diff
        df = n - 1

        # P-value
        if alternative == 'two-sided':
            p_value = 2 * stats.t.cdf(-abs(t_stat), df)
        elif alternative == 'greater':
            p_value = 1 - stats.t.cdf(t_stat, df)
        else:
            p_value = stats.t.cdf(t_stat, df)

        # Effect sizes
        cohen_d = mean_diff / std_diff

        # Correlation between pairs
        correlation = np.corrcoef(before, after)[0, 1]

        # Confidence intervals
        ci_95 = stats.t.interval(0.95, df, loc=mean_diff, scale=se_diff)
        ci_99 = stats.t.interval(0.99, df, loc=mean_diff, scale=se_diff)

        # Wilcoxon signed-rank test as non-parametric alternative
        wilcoxon_stat, wilcoxon_p = stats.wilcoxon(differences, alternative=alternative)

        return {
            't_statistic': t_stat,
            'p_value': p_value,
            'mean_difference': mean_diff,
            'std_difference': std_diff,
            'cohen_d': cohen_d,
            'correlation': correlation,
            'ci_95': ci_95,
            'ci_99': ci_99,
            'wilcoxon_statistic': wilcoxon_stat,
            'wilcoxon_p_value': wilcoxon_p,
            'reject_null': p_value < 0.05
        }

    @staticmethod
    def anova_one_way(
        *groups: np.ndarray,
        post_hoc: bool = True
    ) -> Dict:
        """One-way ANOVA with post-hoc tests and effect size."""

        # Perform ANOVA
        f_stat, p_value = stats.f_oneway(*groups)

        # Calculate effect size (eta-squared)
        all_data = np.concatenate(groups)
        grand_mean = np.mean(all_data)
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
        ss_total = np.sum((all_data - grand_mean)**2)
        eta_squared = ss_between / ss_total

        # Omega-squared (less biased)
        k = len(groups)
        N = len(all_data)
        df_between = k - 1
        df_within = N - k
        ms_between = ss_between / df_between
        ss_within = ss_total - ss_between
        ms_within = ss_within / df_within
        omega_squared = (ss_between - df_between * ms_within) / (ss_total + ms_within)

        result = {
            'f_statistic': f_stat,
            'p_value': p_value,
            'eta_squared': eta_squared,
            'omega_squared': omega_squared,
            'reject_null': p_value < 0.05,
            'group_means': [np.mean(g) for g in groups],
            'group_stds': [np.std(g, ddof=1) for g in groups]
        }

        # Post-hoc tests (Tukey HSD)
        if post_hoc and p_value < 0.05:
            from itertools import combinations
            post_hoc_results = []

            for i, j in combinations(range(len(groups)), 2):
                # Tukey HSD critical value
                q_crit = 3.314  # Approximate for alpha=0.05
                se = np.sqrt(ms_within * (1/len(groups[i]) + 1/len(groups[j])))
                diff = abs(np.mean(groups[i]) - np.mean(groups[j]))
                hsd = q_crit * se

                post_hoc_results.append({
                    'groups': (i, j),
                    'mean_diff': diff,
                    'hsd_threshold': hsd,
                    'significant': diff > hsd
                })

            result['post_hoc'] = post_hoc_results

        return result
```

#### Non-Parametric Tests
```python
class NonParametricHypothesisTesting:
    """Non-parametric hypothesis testing for non-normal distributions."""

    @staticmethod
    def mann_whitney_u(
        sample1: np.ndarray,
        sample2: np.ndarray,
        alternative: str = 'two-sided',
        continuity: bool = True
    ) -> Dict:
        """Mann-Whitney U test with effect size (rank-biserial correlation)."""

        from scipy.stats import mannwhitneyu

        # Perform test
        u_stat, p_value = mannwhitneyu(
            sample1, sample2,
            alternative=alternative,
            use_continuity=continuity
        )

        # Calculate effect size (rank-biserial correlation)
        n1, n2 = len(sample1), len(sample2)
        r_rb = 1 - (2 * u_stat) / (n1 * n2)

        # Cliff's Delta (another effect size)
        comparisons = np.array([[a > b for b in sample2] for a in sample1])
        cliff_delta = np.mean(comparisons) * 2 - 1

        # Hodges-Lehmann estimator (median of differences)
        differences = np.array([a - b for a in sample1 for b in sample2])
        hodges_lehmann = np.median(differences)

        # Bootstrap confidence interval for Hodges-Lehmann
        bootstrap_hl = []
        for _ in range(1000):
            s1_boot = np.random.choice(sample1, size=n1, replace=True)
            s2_boot = np.random.choice(sample2, size=n2, replace=True)
            diff_boot = np.array([a - b for a in s1_boot for b in s2_boot])
            bootstrap_hl.append(np.median(diff_boot))

        ci_95 = np.percentile(bootstrap_hl, [2.5, 97.5])

        return {
            'u_statistic': u_stat,
            'p_value': p_value,
            'rank_biserial': r_rb,
            'cliff_delta': cliff_delta,
            'hodges_lehmann': hodges_lehmann,
            'ci_95_hl': ci_95,
            'reject_null': p_value < 0.05
        }

    @staticmethod
    def kruskal_wallis(
        *groups: np.ndarray,
        post_hoc: bool = True
    ) -> Dict:
        """Kruskal-Wallis H test with post-hoc Dunn test."""

        from scipy.stats import kruskal

        # Perform Kruskal-Wallis test
        h_stat, p_value = kruskal(*groups)

        # Calculate effect size (epsilon-squared)
        all_data = np.concatenate(groups)
        N = len(all_data)
        k = len(groups)
        epsilon_squared = h_stat / (N - 1)

        result = {
            'h_statistic': h_stat,
            'p_value': p_value,
            'epsilon_squared': epsilon_squared,
            'reject_null': p_value < 0.05,
            'group_medians': [np.median(g) for g in groups],
            'group_iqrs': [np.percentile(g, 75) - np.percentile(g, 25) for g in groups]
        }

        # Post-hoc Dunn test
        if post_hoc and p_value < 0.05:
            from itertools import combinations
            post_hoc_results = []

            # Rank all data
            all_ranks = stats.rankdata(all_data)
            group_ranks = []
            idx = 0
            for g in groups:
                group_ranks.append(all_ranks[idx:idx+len(g)])
                idx += len(g)

            for i, j in combinations(range(len(groups)), 2):
                # Calculate z-statistic for Dunn test
                mean_rank_i = np.mean(group_ranks[i])
                mean_rank_j = np.mean(group_ranks[j])
                ni, nj = len(groups[i]), len(groups[j])

                se = np.sqrt((N * (N + 1) / 12) * (1/ni + 1/nj))
                z = abs(mean_rank_i - mean_rank_j) / se

                # Two-tailed p-value
                p = 2 * (1 - stats.norm.cdf(z))

                post_hoc_results.append({
                    'groups': (i, j),
                    'mean_rank_diff': abs(mean_rank_i - mean_rank_j),
                    'z_statistic': z,
                    'p_value': p,
                    'significant': p < 0.05 / len(list(combinations(range(k), 2)))  # Bonferroni
                })

            result['post_hoc'] = post_hoc_results

        return result

    @staticmethod
    def friedman_test(
        *repeated_measures: np.ndarray
    ) -> Dict:
        """Friedman test for repeated measures with Kendall's W."""

        from scipy.stats import friedmanchisquare

        # Check equal lengths
        if len(set(len(m) for m in repeated_measures)) > 1:
            raise ValueError("All repeated measures must have equal length")

        # Perform Friedman test
        chi2_stat, p_value = friedmanchisquare(*repeated_measures)

        # Calculate Kendall's W (concordance coefficient)
        n = len(repeated_measures[0])  # number of subjects
        k = len(repeated_measures)      # number of conditions

        # Rank data for each subject
        ranks = np.array([stats.rankdata(row) for row in np.array(repeated_measures).T])
        mean_ranks = np.mean(ranks, axis=0)

        # Kendall's W
        ss_ranks = np.sum((mean_ranks - np.mean(mean_ranks))**2) * n
        kendall_w = (12 * ss_ranks) / (n**2 * k * (k**2 - 1))

        return {
            'chi2_statistic': chi2_stat,
            'p_value': p_value,
            'kendall_w': kendall_w,
            'mean_ranks': mean_ranks.tolist(),
            'reject_null': p_value < 0.05
        }
```

### 2. Confidence Intervals

```python
class ConfidenceIntervals:
    """Advanced confidence interval calculations."""

    @staticmethod
    def bootstrap_ci(
        data: np.ndarray,
        statistic: callable = np.mean,
        n_bootstrap: int = 10000,
        confidence: float = 0.95,
        method: str = 'percentile'
    ) -> Dict:
        """Bootstrap confidence intervals with multiple methods."""

        # Generate bootstrap samples
        bootstrap_stats = []
        n = len(data)

        for _ in range(n_bootstrap):
            sample = np.random.choice(data, size=n, replace=True)
            bootstrap_stats.append(statistic(sample))

        bootstrap_stats = np.array(bootstrap_stats)
        observed_stat = statistic(data)

        alpha = 1 - confidence
        result = {
            'observed': observed_stat,
            'bootstrap_mean': np.mean(bootstrap_stats),
            'bootstrap_std': np.std(bootstrap_stats)
        }

        if method == 'percentile':
            # Percentile method
            lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
            upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
            result['ci'] = (lower, upper)

        elif method == 'basic':
            # Basic bootstrap
            lower = 2 * observed_stat - np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
            upper = 2 * observed_stat - np.percentile(bootstrap_stats, 100 * alpha / 2)
            result['ci'] = (lower, upper)

        elif method == 'bca':
            # BCa (Bias-Corrected and Accelerated)
            # Calculate bias correction
            z0 = stats.norm.ppf(np.mean(bootstrap_stats < observed_stat))

            # Calculate acceleration
            jackknife_stats = []
            for i in range(n):
                jack_sample = np.delete(data, i)
                jackknife_stats.append(statistic(jack_sample))

            jackknife_mean = np.mean(jackknife_stats)
            numerator = np.sum((jackknife_mean - jackknife_stats)**3)
            denominator = 6 * np.sum((jackknife_mean - jackknife_stats)**2)**(3/2)
            acceleration = numerator / denominator if denominator != 0 else 0

            # Adjusted percentiles
            z_alpha = stats.norm.ppf(alpha / 2)
            z_1alpha = stats.norm.ppf(1 - alpha / 2)

            p_lower = stats.norm.cdf(z0 + (z0 + z_alpha) / (1 - acceleration * (z0 + z_alpha)))
            p_upper = stats.norm.cdf(z0 + (z0 + z_1alpha) / (1 - acceleration * (z0 + z_1alpha)))

            lower = np.percentile(bootstrap_stats, 100 * p_lower)
            upper = np.percentile(bootstrap_stats, 100 * p_upper)
            result['ci'] = (lower, upper)
            result['bias_correction'] = z0
            result['acceleration'] = acceleration

        return result

    @staticmethod
    def wilson_score_interval(
        successes: int,
        trials: int,
        confidence: float = 0.95
    ) -> Tuple[float, float]:
        """Wilson score interval for binomial proportions."""

        if trials == 0:
            return (0.0, 0.0)

        p_hat = successes / trials
        z = stats.norm.ppf(1 - (1 - confidence) / 2)
        z2 = z * z

        denominator = 1 + z2 / trials
        center = (p_hat + z2 / (2 * trials)) / denominator
        spread = z * np.sqrt(p_hat * (1 - p_hat) / trials + z2 / (4 * trials**2)) / denominator

        return (max(0, center - spread), min(1, center + spread))

    @staticmethod
    def difference_of_proportions_ci(
        successes1: int,
        trials1: int,
        successes2: int,
        trials2: int,
        confidence: float = 0.95
    ) -> Dict:
        """Confidence interval for difference of two proportions."""

        p1 = successes1 / trials1
        p2 = successes2 / trials2
        diff = p1 - p2

        # Standard error
        se = np.sqrt(p1 * (1 - p1) / trials1 + p2 * (1 - p2) / trials2)

        # Z-score
        z = stats.norm.ppf(1 - (1 - confidence) / 2)

        # Wald interval
        wald_ci = (diff - z * se, diff + z * se)

        # Agresti-Caffo interval (add 1 success and 1 failure to each group)
        p1_ac = (successes1 + 1) / (trials1 + 2)
        p2_ac = (successes2 + 1) / (trials2 + 2)
        diff_ac = p1_ac - p2_ac
        se_ac = np.sqrt(p1_ac * (1 - p1_ac) / (trials1 + 2) + p2_ac * (1 - p2_ac) / (trials2 + 2))
        ac_ci = (diff_ac - z * se_ac, diff_ac + z * se_ac)

        return {
            'difference': diff,
            'wald_ci': wald_ci,
            'agresti_caffo_ci': ac_ci,
            'p1': p1,
            'p2': p2,
            'se': se
        }
```

### 3. Multiple Testing Corrections

```python
class MultipleTestingCorrections:
    """Comprehensive multiple testing correction methods."""

    @staticmethod
    def bonferroni(
        p_values: np.ndarray,
        alpha: float = 0.05
    ) -> Dict:
        """Bonferroni correction for family-wise error rate."""

        n = len(p_values)
        adjusted_alpha = alpha / n
        adjusted_p_values = np.minimum(p_values * n, 1.0)
        rejected = adjusted_p_values < alpha

        return {
            'method': 'Bonferroni',
            'adjusted_alpha': adjusted_alpha,
            'adjusted_p_values': adjusted_p_values,
            'rejected': rejected,
            'num_rejected': np.sum(rejected)
        }

    @staticmethod
    def holm_bonferroni(
        p_values: np.ndarray,
        alpha: float = 0.05
    ) -> Dict:
        """Holm-Bonferroni step-down procedure."""

        n = len(p_values)
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]

        adjusted_p = np.zeros(n)
        rejected = np.zeros(n, dtype=bool)

        for i in range(n):
            adjusted_p[i] = min(sorted_p[i] * (n - i), 1.0)
            if i > 0:
                adjusted_p[i] = max(adjusted_p[i], adjusted_p[i-1])

            if adjusted_p[i] < alpha:
                rejected[sorted_idx[i]] = True
            else:
                break  # Stop at first non-rejection

        # Reorder to original positions
        adjusted_p_original = np.zeros(n)
        for i, idx in enumerate(sorted_idx):
            adjusted_p_original[idx] = adjusted_p[i]

        return {
            'method': 'Holm-Bonferroni',
            'adjusted_p_values': adjusted_p_original,
            'rejected': rejected,
            'num_rejected': np.sum(rejected)
        }

    @staticmethod
    def benjamini_hochberg(
        p_values: np.ndarray,
        alpha: float = 0.05
    ) -> Dict:
        """Benjamini-Hochberg FDR control."""

        n = len(p_values)
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]

        # Find largest i such that P(i) <= (i/n) * alpha
        rejected = np.zeros(n, dtype=bool)
        threshold_idx = -1

        for i in range(n - 1, -1, -1):
            if sorted_p[i] <= (i + 1) / n * alpha:
                threshold_idx = i
                break

        if threshold_idx >= 0:
            rejected[sorted_idx[:threshold_idx + 1]] = True

        # Calculate adjusted p-values
        adjusted_p = np.zeros(n)
        for i in range(n):
            adjusted_p[i] = min(sorted_p[i] * n / (i + 1), 1.0)

        # Enforce monotonicity
        for i in range(n - 2, -1, -1):
            adjusted_p[i] = min(adjusted_p[i], adjusted_p[i + 1])

        # Reorder to original positions
        adjusted_p_original = np.zeros(n)
        for i, idx in enumerate(sorted_idx):
            adjusted_p_original[idx] = adjusted_p[i]

        return {
            'method': 'Benjamini-Hochberg',
            'adjusted_p_values': adjusted_p_original,
            'rejected': rejected,
            'num_rejected': np.sum(rejected),
            'fdr_threshold': sorted_p[threshold_idx] if threshold_idx >= 0 else None
        }

    @staticmethod
    def benjamini_yekutieli(
        p_values: np.ndarray,
        alpha: float = 0.05
    ) -> Dict:
        """Benjamini-Yekutieli FDR control (works under dependence)."""

        n = len(p_values)
        c = np.sum(1 / np.arange(1, n + 1))  # Harmonic sum
        adjusted_alpha = alpha / c

        # Apply BH procedure with adjusted alpha
        return MultipleTestingCorrections.benjamini_hochberg(p_values, adjusted_alpha)

    @staticmethod
    def storey_q_value(
        p_values: np.ndarray,
        pi0: Optional[float] = None,
        lambda_: float = 0.5
    ) -> Dict:
        """Storey's q-value with automatic pi0 estimation."""

        n = len(p_values)
        sorted_idx = np.argsort(p_values)
        sorted_p = p_values[sorted_idx]

        # Estimate pi0 if not provided
        if pi0 is None:
            # Storey's estimator
            W = np.sum(p_values > lambda_)
            pi0 = min(1, W / (n * (1 - lambda_)))

        # Calculate q-values
        q_values = np.zeros(n)
        for i in range(n):
            q_values[i] = sorted_p[i] * n * pi0 / (i + 1)

        # Enforce monotonicity
        for i in range(n - 2, -1, -1):
            q_values[i] = min(q_values[i], q_values[i + 1])

        # Reorder to original positions
        q_values_original = np.zeros(n)
        for i, idx in enumerate(sorted_idx):
            q_values_original[idx] = q_values[i]

        return {
            'method': 'Storey q-value',
            'q_values': q_values_original,
            'pi0': pi0,
            'significant': q_values_original < 0.05,
            'num_significant': np.sum(q_values_original < 0.05)
        }
```

### 4. Scientific Writing Templates

```python
class ScientificWritingTemplates:
    """Templates for scientific paper sections."""

    @staticmethod
    def generate_abstract(
        problem: str,
        method: str,
        results: str,
        implications: str,
        max_words: int = 250
    ) -> str:
        """Generate structured abstract."""

        template = f"""
ABSTRACT

{problem} In this work, we propose {method}. Our approach differs from prior work
by [KEY INNOVATION]. Experiments on [DATASETS] demonstrate that our method {results},
achieving [X]% improvement over state-of-the-art baselines. Statistical analysis
confirms significance (p < 0.001) with large effect sizes (Cohen's d > 0.8).
{implications} Code and data are available at [URL].

Word count: [CALCULATE]
"""
        return template

    @staticmethod
    def generate_introduction_outline() -> str:
        """Generate introduction section outline."""

        return """
## 1. Introduction

### Paragraph 1: Problem Statement
- What is the general area?
- Why is it important?
- What are the current limitations?

### Paragraph 2: Existing Approaches
- How have others tried to solve this?
- What are the strengths and weaknesses?
- What gap remains?

### Paragraph 3: Our Approach
- What is our key insight?
- How does our method work at a high level?
- Why should it work better?

### Paragraph 4: Contributions
We make the following contributions:
1. **Novel Method**: [Description]
2. **Theoretical Analysis**: [Description]
3. **Empirical Validation**: [Description]
4. **Open Source**: [Description]

### Paragraph 5: Paper Organization
The remainder of this paper is organized as follows: Section 2 reviews related work,
Section 3 describes our methodology, Section 4 presents experimental results,
Section 5 discusses implications and limitations, and Section 6 concludes.
"""

    @staticmethod
    def generate_results_table(
        results: Dict[str, Dict[str, float]],
        metrics: List[str],
        highlight_best: bool = True
    ) -> str:
        """Generate LaTeX results table."""

        # Find best scores
        best_scores = {}
        for metric in metrics:
            scores = [results[method].get(metric, 0) for method in results]
            best_scores[metric] = max(scores)

        # Generate LaTeX
        latex = "\\begin{table}[htbp]\n\\centering\n"
        latex += "\\caption{Experimental results. Best scores are \\textbf{bolded}.}\n"
        latex += "\\label{tab:results}\n"
        latex += "\\begin{tabular}{l" + "c" * len(metrics) + "}\n"
        latex += "\\toprule\n"
        latex += "Method & " + " & ".join(metrics) + " \\\\\n"
        latex += "\\midrule\n"

        for method in results:
            row = method.replace("_", "\\_")
            for metric in metrics:
                score = results[method].get(metric, 0)
                if highlight_best and score == best_scores[metric]:
                    row += f" & \\textbf{{{score:.3f}}}"
                else:
                    row += f" & {score:.3f}"
            row += " \\\\\n"
            latex += row

        latex += "\\bottomrule\n"
        latex += "\\end{tabular}\n"
        latex += "\\end{table}"

        return latex

    @staticmethod
    def generate_discussion_points() -> List[str]:
        """Generate discussion section talking points."""

        return [
            "### Key Findings",
            "- What was most surprising?",
            "- What confirmed expectations?",
            "- What patterns emerged?",

            "### Theoretical Implications",
            "- How do results relate to theory?",
            "- What assumptions were validated/invalidated?",
            "- What new questions arise?",

            "### Practical Applications",
            "- Who would benefit from this?",
            "- What are the deployment considerations?",
            "- What is the computational cost?",

            "### Limitations",
            "- Dataset limitations",
            "- Methodological constraints",
            "- Generalization boundaries",
            "- Computational requirements",

            "### Future Work",
            "- Immediate extensions",
            "- Long-term research directions",
            "- Open problems"
        ]

    @staticmethod
    def format_statistical_result(
        test_name: str,
        statistic: float,
        p_value: float,
        effect_size: float,
        effect_name: str = "d"
    ) -> str:
        """Format statistical results for publication."""

        # Format p-value
        if p_value < 0.001:
            p_str = "p < 0.001"
        elif p_value < 0.01:
            p_str = "p < 0.01"
        elif p_value < 0.05:
            p_str = "p < 0.05"
        else:
            p_str = f"p = {p_value:.3f}"

        # Interpret effect size
        if effect_name == "d":  # Cohen's d
            if abs(effect_size) < 0.2:
                interpretation = "negligible"
            elif abs(effect_size) < 0.5:
                interpretation = "small"
            elif abs(effect_size) < 0.8:
                interpretation = "medium"
            else:
                interpretation = "large"
        else:
            interpretation = ""

        result = f"{test_name}: statistic = {statistic:.3f}, {p_str}, "
        result += f"{effect_name} = {effect_size:.3f}"
        if interpretation:
            result += f" ({interpretation} effect)"

        return result
```

### 5. Reproducibility Framework

```python
import json
import hashlib
from datetime import datetime
from pathlib import Path

class ReproducibilityFramework:
    """Comprehensive reproducibility management system."""

    def __init__(self, experiment_name: str, base_dir: str = "./experiments"):
        self.experiment_name = experiment_name
        self.base_dir = Path(base_dir)
        self.exp_dir = self.base_dir / experiment_name
        self.exp_dir.mkdir(parents=True, exist_ok=True)

        self.config = {
            'experiment_name': experiment_name,
            'timestamp': datetime.now().isoformat(),
            'environment': {},
            'hyperparameters': {},
            'data': {},
            'results': {}
        }

    def log_environment(self):
        """Log complete environment information."""

        import platform
        import sys
        import torch
        import numpy as np

        self.config['environment'] = {
            'python_version': sys.version,
            'platform': platform.platform(),
            'processor': platform.processor(),
            'hostname': platform.node(),
            'torch_version': torch.__version__,
            'numpy_version': np.__version__,
            'cuda_available': torch.cuda.is_available(),
            'cuda_version': torch.version.cuda if torch.cuda.is_available() else None,
            'gpu_count': torch.cuda.device_count() if torch.cuda.is_available() else 0,
            'gpu_names': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]
                        if torch.cuda.is_available() else []
        }

    def log_data_fingerprint(self, data_path: str):
        """Create reproducible data fingerprint."""

        # Calculate file hash
        hasher = hashlib.sha256()
        with open(data_path, 'rb') as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hasher.update(chunk)

        self.config['data']['path'] = str(data_path)
        self.config['data']['sha256'] = hasher.hexdigest()
        self.config['data']['size_bytes'] = Path(data_path).stat().st_size

    def set_global_seed(self, seed: int = 42):
        """Set all random seeds for reproducibility."""

        import random
        import numpy as np
        import torch

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        # Deterministic operations
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        # Environment variable
        import os
        os.environ['PYTHONHASHSEED'] = str(seed)

        self.config['environment']['random_seed'] = seed
        self.config['environment']['deterministic'] = True

    def log_hyperparameters(self, **kwargs):
        """Log all hyperparameters."""
        self.config['hyperparameters'].update(kwargs)

    def log_results(self, **kwargs):
        """Log experimental results."""
        self.config['results'].update(kwargs)

    def save_checkpoint(self, model, optimizer, epoch, metrics):
        """Save training checkpoint."""

        checkpoint_path = self.exp_dir / f"checkpoint_epoch_{epoch}.pt"

        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics,
            'config': self.config
        }, checkpoint_path)

        return checkpoint_path

    def save_config(self):
        """Save complete configuration."""

        config_path = self.exp_dir / "config.json"
        with open(config_path, 'w') as f:
            json.dump(self.config, f, indent=2, default=str)

        return config_path

    def create_requirements_file(self):
        """Generate requirements.txt file."""

        import pkg_resources

        requirements = []
        for dist in pkg_resources.working_set:
            requirements.append(f"{dist.key}=={dist.version}")

        req_path = self.exp_dir / "requirements.txt"
        with open(req_path, 'w') as f:
            f.write('\n'.join(sorted(requirements)))

        return req_path

    def create_reproducibility_report(self) -> str:
        """Generate comprehensive reproducibility report."""

        report = f"""
# Reproducibility Report: {self.experiment_name}

Generated: {datetime.now().isoformat()}

## Environment
- Python: {self.config['environment'].get('python_version', 'N/A')}
- PyTorch: {self.config['environment'].get('torch_version', 'N/A')}
- CUDA: {self.config['environment'].get('cuda_version', 'N/A')}
- Random Seed: {self.config['environment'].get('random_seed', 'N/A')}

## Data
- Path: {self.config['data'].get('path', 'N/A')}
- SHA256: {self.config['data'].get('sha256', 'N/A')}
- Size: {self.config['data'].get('size_bytes', 0) / 1024**2:.2f} MB

## Hyperparameters
"""
        for key, value in self.config['hyperparameters'].items():
            report += f"- {key}: {value}\n"

        report += "\n## Results\n"
        for key, value in self.config['results'].items():
            report += f"- {key}: {value}\n"

        report += "\n## Reproduction Steps\n"
        report += """
1. Install dependencies: `pip install -r requirements.txt`
2. Set random seed: Use provided seed configuration
3. Load data: Verify SHA256 matches
4. Run training: Use exact hyperparameters
5. Evaluate: Follow evaluation protocol
"""

        return report
```

This skill provides comprehensive capabilities for hypothesis testing, confidence intervals, multiple testing corrections (FDR/Bonferroni), and scientific writing templates. It includes both parametric and non-parametric tests, various effect size measures, bootstrap methods, and complete reproducibility frameworks for ensuring research validity and transparency.