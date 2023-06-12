from typing import Tuple
from execution_time_func_decorator import timeit
import numpy as np
import pandas as pd

#### Boostrapping ####
# scipy's alternative: 
# from scipy.stats import bootstrap

def _bootstrap_statistic_sampling_distribution(
    series: pd.Series,
    statistic: callable,
    nrepeat: int) -> pd.Series:
    """
    Function that returns the sampling distribution of a statistic via boostrapping (sampling with replacement).

    Args:
        series (pd.Series): Input data as pandas Series (column of a df)
        statistic  (callable): Statistic as function that can be called (eg np.mean).
        nrepeat (int): Number of iterations.

    Returns:
        pd.Series: A pandas Series containing the bootstrapped sampling distribution for the chosen statistic.
    """

    results = []
    # start boostrap
    for _ in range(nrepeat):
        sample = series.sample(n=series.shape[0], replace=True)
        results.append(statistic(sample))
    results = pd.Series(results)

    return results

@timeit
def bootstrap_statistic_std_error(
    df:pd.DataFrame, 
    column_name:str, 
    statistic:callable, 
    nrepeat:int = 5000) -> Tuple[float, float, float]:
    """
    Function to estimate the sampling distribution of a statistic via boostrapping,
    and compute bias (vs original estimate) and the standard error.

    Args:
        df (pd.DataFrame): Input data as dataframe
        column_name (str): Name of column whose statistic wants to be analysed
        statistic (callable): Statistic as function that can be called (eg np.mean).
        nrepeat (int, optional): Number of iterations. Defaults to 5000.

    Returns:
        Tuple: A tuple containing the original estimate of the statistic,
        the bias of original estimate vs bootstrapped estimate, 
        and the bootstrapped standard error of the statistic.
    """

    # start boostrap
    series = df[column_name]
    results = _bootstrap_statistic_sampling_distribution(series, statistic, nrepeat)
    
    # compute & return statistics
    original_estimate = statistic(series)
    bootstrapped_estimate = results.mean()
    bias = original_estimate - bootstrapped_estimate #TODO double check if it should rather be the other way around? 
    std_error = results.std()

    return original_estimate, bias, std_error

@timeit
def boostrap_statistic_with_confidence_interval(
        df: pd.DataFrame,
        column_name: str,
        statistic: callable,
        confidence_level: float = 0.95,
        nrepeat: int = 1000) -> Tuple[float, float, float]:
    """
    Function to estimate the estimate of a statistic via boostrapping,
    and compute the confidence interval of this estimate at required confidence_level.

    Args:
        df (pd.DataFrame): Input data as dataframe
        column_name (str): Name of column whose statistic wants to be analysed
        confidence_level (float, optional): Required confidence level. Defaults to 0.95.
        statistic (callable): Statistic as function that can be called (eg np.mean).
        nrepeat (int, optional): Number of iterations. Defaults to 1000.

    Returns:
        Tuple: A tuple containing the bootstrapped estimate of the statistic 
        and the lower & upper bound (respectively in this order) of the confidence interval.
    """

    # start boostrap
    series = df[column_name]
    results = _bootstrap_statistic_sampling_distribution(
        series, statistic, nrepeat)

    # compute estimate & CI
    bootstrapped_estimate = results.mean()
    alpha = 1 - confidence_level
    lower_quantile = alpha / 2
    upper_quantile = 1 - (alpha / 2)
    lower_bound = results.quantile(q=lower_quantile)
    upper_bound = results.quantile(q=upper_quantile)

    return bootstrapped_estimate, lower_bound, upper_bound


#### Permutation tests ####
# scipy's alternative: 
# from scipy.stats import permutation_test

# representative image ex.: https://miro.medium.com/v2/resize:fit:828/format:webp/1*n0zLfPro2BYdku9z3LqNqw.png

def _permutation_difference_statistic_two_groups(
        x: pd.Series,
        n_b: int) -> float :
    """Given 2 groups A and B, combines their results via permutation
    and calculates the difference in mean (represents an iteration of the null hypothesis)

    Args:
        x (pd.Series): Series with outcome values
        n_b (int): Number of observations for group B

    Returns:
        float: Difference in mean after applying a permutation between the 2 groups.
    """

    n = x.shape[0]
    rng = np.random.default_rng()
    idx_b = set(rng.choice(np.arange(n), size=n_b, replace=False))
    idx_a = set(range(n)) - idx_b
    result = x.loc[list(idx_b)].mean() - x.loc[list(idx_a)].mean()

    return result

@timeit
def permutation_test_statistic_two_groups(
        df: pd.DataFrame,
        output_col_name: str,
        groups_col_name: str,
        treatment_group_name: str,
        control_group_name: str,
        nrepeat: int = 5000) -> Tuple[float, float]:
    """Computes mean difference between two groups and p_value vs null hypothesis (groups being equal).

    Args:
        df (pd.DataFrame): Input dataframe
        output_col_name (str): Name of output column
        groups_col_name (str): Name of column with groups
        treatment_group_name (str): Name of the treatment group
        control_group_name (str): Name of the control group
        nrepeat (int, optional): Number of permutations. Defaults to 5000.

    Returns:
        Tuple: A tuple containing the observed mean difference between the 2 groups (treatment - control) 
        and the p_value vs the null hypothesis (groups being equal).
    """

    x_groups = df[df[groups_col_name].isin([treatment_group_name, control_group_name])].reset_index()
    x = x_groups[output_col_name]
    group_a = control_group_name
    group_b = treatment_group_name
    n_b = df[df[groups_col_name] == group_b].shape[0]

    mean_a = df.loc[df[groups_col_name] == group_a, [output_col_name]].mean()
    mean_b = df.loc[df[groups_col_name] == group_b, [output_col_name]].mean()
    observed_difference = (mean_b - mean_a).item()
    # perm_diffs = [_permutation_difference_statistic_two_groups(
    #     x, n, n_b) for _ in range(nrepeat)]
    # p_value = np.mean([diff > observed_difference for diff in perm_diffs])
    perm_diffs = np.fromiter((_permutation_difference_statistic_two_groups(
        x, n_b) for _ in range(nrepeat)), dtype=float)
    p_value = np.mean(perm_diffs > abs(observed_difference))

    return observed_difference, p_value

@timeit
def optimised_permutation_test_statistic_two_groups(
        df: pd.DataFrame,
        output_col_name: str,
        groups_col_name: str,
        treatment_group_name: str,
        control_group_name: str,
        nrepeat: int = 5000) -> Tuple[float, float]:
    """
    Computes mean difference between two groups and p_value vs null hypothesis (groups being equal).
    Optimised by vectorising functions instead of using a list comprehension (50x faster).

    Args:
        df (pd.DataFrame): Input dataframe
        output_col_name (str): Name of output column
        groups_col_name (str): Name of column with groups
        treatment_group_name (str): Name of the treatment group
        control_group_name (str): Name of the control group
        nrepeat (int, optional): Number of permutations. Defaults to 5000.

    Returns:
        Tuple: A tuple containing the observed mean difference between the 2 groups (treatment - control) 
        and the p_value vs the null hypothesis (groups being equal).
    """

    x_groups = df[df[groups_col_name].isin([treatment_group_name, control_group_name])].reset_index()
    x = x_groups[output_col_name]
    group_a = control_group_name
    group_b = treatment_group_name
    n = x.shape[0]
    n_b = df[df[groups_col_name] == group_b].shape[0]

    mean_a = df.loc[df[groups_col_name] == group_a, [output_col_name]].mean()
    mean_b = df.loc[df[groups_col_name] == group_b, [output_col_name]].mean()
    observed_difference = (mean_b - mean_a).item()

    # optimised part
    values = np.tile(x, (nrepeat,1))
    mask = np.arange(n) * np.ones(shape=(nrepeat, n)) >= n_b
    np.apply_along_axis(func1d=np.random.shuffle, axis=1, arr=mask)
    group_a_values = values.flatten()[mask.flatten()].reshape(nrepeat, n - n_b)
    group_b_values = values.flatten()[~mask.flatten()].reshape(nrepeat, n_b)

    perm_diffs = group_a_values.mean(axis=1) - group_b_values.mean(axis=1)
    p_value = np.mean(perm_diffs > abs(observed_difference))

    return observed_difference, p_value


if __name__ == "__main__":
    # try out
    titanic = pd.read_csv(
        'https://storage.googleapis.com/tf-datasets/titanic/train.csv')
    observed_diff, pvalue = optimised_permutation_test_statistic_two_groups(df=titanic, output_col_name="n_siblings_spouses", 
                                                                  groups_col_name="class", treatment_group_name="First", 
                                                                  control_group_name="Second",  nrepeat=5000)
    print(f"Observed difference (treatment - control): {observed_diff} \np-value: {pvalue}")
