#!/usr/bin/env python
# coding: utf-8

# Q1:
#     The standard deviation measures the spread or dispersion of the individual data points in a data set relative to the mean of that data set. It tells you how much the data points in a sample or population typically deviate from the mean. A larger standard deviation means that the data points are more spread out, while a smaller standard deviation means that the data points are more concentrated around the mean. The standard error of the mean estimates how much the sample mean (the average of the samples drawn from the population) would be expected to vary if different samples were drawn from the population.
# It indicates how accurately the sample mean estimates the population mean. A smaller SEM means that the sample mean is a more accurate estimate of the population mean. The SEM decreases as the sample size increases. The standard error of the mean (SEM) is calculated by dividing the standard deviation (SD) by the square root of the sample size: This formula shows that the SEM is directly related to the standard deviation and the sample size. As the sample size increases, the SEM decreases. This means that when the sample size is larger, our estimate of the sample mean as a population mean will be more accurate.

# Q2：
# Estimation of 95% confidence interval:
# 
# Based on the assumption of normal distribution: Under normal distribution, about 95% of the data will fall within 1.96 times the standard error of the mean. Usually, for the purpose of simplicity, 2×SEM is used to replace 1.96 times SEM.
# Explanation: This interval means that if we sample from the population multiple times and calculate the sample mean, 95% of the sample means will fall within this interval. This method is simple and easy, but it assumes that the data is normally distributed or approximately normally distributed.

# Q3：
#     1. Repeatedly draw from the original sample (with replacement) through bootstrapped, each time generating a sample of the same size as the original sample. 
#     2. For each bootstrapped sample, calculate its mean, usually repeated thousands of times, to construct the distribution of the sample mean.
#     3.Based on the distribution of the bootstrapped sample mean, select the 2.5% and 97.5% percentiles as the lower and upper limits of the 95% confidence interval.

# In[3]:


import numpy as np

data = np.random.randn(100)  


n_iterations = 10000  
n_size = len(data)  
means = []


for _ in range(n_iterations):

    sample = np.random.choice(data, size=n_size, replace=True)

    sample_mean = np.mean(sample)
    means.append(sample_mean)


lower_bound = np.percentile(means, 2.5)  
upper_bound = np.percentile(means, 97.5)  


print(f"95%CI: ({lower_bound}, {upper_bound})")


# In[6]:


#Q4
import numpy as np

# Define the function to compute bootstrap confidence intervals
def bootstrap_confidence_interval(data, statistic_func, n_bootstrap=10000, ci=95):
    """
    Calculate a bootstrap confidence interval for a given statistic.
    
    Parameters:
    data (array-like): The sample data.
    statistic_func (function): The statistic function to apply (e.g., np.mean, np.median).
    n_bootstrap (int): Number of bootstrap samples (default=10000).
    ci (float): Confidence interval percentage (default=95).
    
    Returns:
    tuple: Lower and upper bounds of the confidence interval.
    """
    bootstrap_statistics = []
    n_size = len(data)
    
    # Bootstrap resampling process
    for _ in range(n_bootstrap):
        # Random sampling with replacement
        bootstrap_sample = np.random.choice(data, size=n_size, replace=True)
        # Apply the statistic function (e.g., mean or median) to the bootstrap sample
        statistic_value = statistic_func(bootstrap_sample)
        bootstrap_statistics.append(statistic_value)
    
    # Calculate confidence interval bounds
    lower_bound = np.percentile(bootstrap_statistics, (100 - ci) / 2)
    upper_bound = np.percentile(bootstrap_statistics, 100 - (100 - ci) / 2)
    
    return lower_bound, upper_bound

# Example usage with your data
data = np.array([12, 15, 14, 10, 13, 14, 16, 13, 11, 12])  # Example data

# Bootstrap confidence interval for the mean
lower_ci_mean, upper_ci_mean = bootstrap_confidence_interval(data, np.mean, n_bootstrap=10000, ci=95)
print(f"Bootstrap 95% Confidence Interval for the Mean: ({lower_ci_mean}, {upper_ci_mean})")

# Bootstrap confidence interval for the median
lower_ci_median, upper_ci_median = bootstrap_confidence_interval(data, np.median, n_bootstrap=10000, ci=95)
print(f"Bootstrap 95% Confidence Interval for the Median: ({lower_ci_median}, {upper_ci_median})")


# Q5:
# 1. Sample statistics are random, and population parameters are fixed:
# 
# Population parameters are fixed because they describe the characteristics of a population. No matter how we draw samples, the population mean, population standard deviation, etc. are fixed values.
# Sample statistics are random because the samples obtained each time may be different, so sample statistics (such as sample mean, sample median) will also change. This is why we need to estimate population parameters through sample statistics.
# For example, if we draw samples from the same population multiple times, the mean of each sample may be different, but we hope to infer the population mean through these sample means.
# 
# 2. Randomness of confidence intervals:
# 
# Confidence intervals are not a fixed range, but an estimated interval calculated based on samples. Therefore, the range of the confidence interval depends on the sample statistic, not the population parameter.
# For example, the meaning of a 95% confidence interval is: if we repeat sampling and calculating confidence intervals multiple times, then 95% of the confidence intervals will contain the population parameter.

# Q6：
# Resample with replacement: From the original sample, draw repeated samples of the same size (called "bootstrap samples") with replacement.
# Calculate statistics: For each bootstrap sample, calculate the statistic of interest (e.g., mean, median).
# Repeat: Repeat the resampling and calculation process many times (usually thousands of times).
# Create distribution: This produces a distribution of the statistic calculated from the bootstrap samples.
# Estimate confidence intervals: Calculate confidence intervals for the statistic from the bootstrap distribution using percentiles or other methods.
# Main purpose of bootstrapping:
# The main purpose of the bootstrap method is to estimate the sampling distribution of a statistic (such as the mean or median) when the theoretical distribution is unknown or difficult to derive. It helps create confidence intervals and test assumptions about population parameters without relying on strong parametric assumptions.
# 
# Evaluate a hypothesized population mean using bootstrapping:
# Resample the original sample using bootstrapping to create many bootstrap samples.
# Calculate the mean of each bootstrap sample.
# Generate a distribution of the bootstrap sample means.
# Compare the hypothesized population mean to the bootstrap distribution.
# If the hypothesized mean falls within the confidence interval of the bootstrap mean, then it is reasonable.
# If it is outside the confidence interval, it may indicate that the hypothesis is not very plausible.

# Q7:
# Inclusive zero between Okishin area: Possible limit for displaying certain specific measurements (for example, Hitoshi Yoshimoto) within Okishin area. As a result, the scope of the belief system is zero, and the meaning can be imprinted with no actual effect (immediate average effect is zero). Because of this, there are many possibilities for me to legally exclude myself.
# 
# Zero-rejection: Zero-reference (zero-reference) Normally, the effect or effect of the design is different. At this time, when we place a message in the area, there is no meaning attached to it, so it is a reasonable solution. Because of this, there was no confirmation that the product had any effect, so it was illegal to refuse.
# 
# Is it “refusal to refuse” to lead a party?
# As a result, the actual effect of the actual object is zero, and the average effect of the actual object is not zero. This is the average result of the product production and its impact. At this time, there is no rational solution, so we refuse to accept it.
# 
# Summary: The inclusion of zero meaning: the existence of an effective possibility without any meaning, and the possibility of an impossibility.
# The meaning of the emblem is the same, the meaning of the emblem is effective, the exclusion of the emblem is effective, and the embedding is confirmed to be effective.

# #Q8:
# Introduction
# AliTech has developed a vaccine that is intended to improve the health of those who receive it. In this case,
# the null hypothesis is that the vaccine has no effect on health scores, meaning that the average change in health
# scores before and after vaccination is zero. Our task is to analyze the data to determine if we can reject this 
# null hypothesis and conclude that the vaccine is effective.
# Null Hypothesis Explanation:
# In this analysis, the null hypothesis states that the vaccine has no effect on health outcomes. In statistical terms, this means that the average change between the initial and final health scores is zero. If the confidence interval for the change in health scores includes zero, we fail to reject the null hypothesis, indicating that there is no significant evidence to suggest the vaccine works.

# In[7]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Setting seed for reproducibility
np.random.seed(42)

# Creating a DataFrame from the provided data
data = {
    'PatientID': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Age': [45, 34, 29, 52, 37, 41, 33, 48, 26, 39],
    'Gender': ['M', 'F', 'M', 'F', 'M', 'F', 'M', 'F', 'M', 'F'],
    'InitialHealthScore': [84, 78, 83, 81, 81, 80, 79, 85, 76, 83],
    'FinalHealthScore': [86, 86, 80, 86, 84, 86, 86, 82, 83, 84]
}
df = pd.DataFrame(data)

# Step 1: Data Visualization
# Showing the distribution of Initial and Final Health Scores
plt.figure(figsize=(10,6))
sns.boxplot(data=df[['InitialHealthScore', 'FinalHealthScore']])
plt.title('Comparison of Initial and Final Health Scores')
plt.ylabel('Health Score')
plt.xticks([0, 1], ['Initial Health Score', 'Final Health Score'])
plt.show()

# Step 2: Quantitative Analysis - Using bootstrapping to assess the effect
def bootstrap_confidence_interval(data, statistic_func, n_bootstrap=10000, ci=95):
    """
    Calculate a bootstrap confidence interval for a given statistic.
    
    Parameters:
    data (array-like): The sample data.
    statistic_func (function): The statistic function to apply (e.g., np.mean).
    n_bootstrap (int): Number of bootstrap samples (default=10000).
    ci (float): Confidence interval percentage (default=95).
    
    Returns:
    tuple: Lower and upper bounds of the confidence interval.
    """
    bootstrap_statistics = []
    n_size = len(data)
    
    # Bootstrap resampling process
    for _ in range(n_bootstrap):
        bootstrap_sample = np.random.choice(data, size=n_size, replace=True)
        statistic_value = statistic_func(bootstrap_sample)
        bootstrap_statistics.append(statistic_value)
    
    # Calculate confidence interval bounds
    lower_bound = np.percentile(bootstrap_statistics, (100 - ci) / 2)
    upper_bound = np.percentile(bootstrap_statistics, 100 - (100 - ci) / 2)
    
    return lower_bound, upper_bound

# Computing the difference in health scores
df['HealthScoreChange'] = df['FinalHealthScore'] - df['InitialHealthScore']

# Perform bootstrapping for the mean health score change
lower_ci, upper_ci = bootstrap_confidence_interval(df['HealthScoreChange'], np.mean, n_bootstrap=10000, ci=95)

# Displaying the 95% confidence interval for health score changes
lower_ci, upper_ci


# The boxplot reveals the spread and central tendency of both Initial and Final Health Scores, which provides an initial visual insight into potential improvements after taking the vaccine.

# Quantitative Analysis: Methodology
# We used bootstrapping to assess whether the vaccine leads to a significant improvement in health scores. Bootstrapping allows us to resample the data with replacement multiple times and estimate a confidence interval for the mean change in health scores.
# 
# We computed the difference between the final health scores and initial health scores for each patient.
# Bootstrapping was used to generate 10,000 resamples of this difference, and a 95% confidence interval was calculated based on the resampling distribution.
# 

# Results:
# The 95% confidence interval for the mean health score change is as follows:
# 
# Lower bound: 1.1
# Upper bound: 5.1
# Further considerations
# The sample size was relatively small (10 patients), which may limit the robustness of the findings.
# Future studies could benefit from larger sample sizes or more detailed health indicators to improve the reliability of the results.

# ChatGPT Link: https://chatgpt.com/share/66fea90b-2180-8006-939a-fb483587833f

# Q9: SOMEWHAT
