import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from itertools import combinations

class CategoricalEDA:
    def __init__(self, df: pd.DataFrame):
        """
        Initializes the EDA class with the dataset.
        
        Args:
        df (pd.DataFrame): The dataset to be analyzed.
        """
        self.df = df
    
    def frequency_counts(self, column: str):
        """
        Returns the frequency counts of a categorical variable.
        
        Args:
        column (str): Column name of the categorical variable.
        
        Returns:
        pd.Series: Frequency counts of each category.
        """
        return self.df[column].value_counts()

    def plot_bar_chart(self, column: str, title: str = None):
        """
        Plots a bar chart of the categorical variable and adds percentages.
        
        Args:
        column (str): Column name of the categorical variable.
        title (str): Title of the chart (optional).
        """
        counts = self.frequency_counts(column)
        total = counts.sum()  # Total count for percentage calculation
        
        plt.figure(figsize=(10, 6))
        ax = sns.barplot(x=counts.index, y=counts.values)
        
        # Annotate bars with percentages
        for p in ax.patches:
            percentage = f'{100 * p.get_height() / total:.1f}%'
            ax.annotate(percentage, 
                        (p.get_x() + p.get_width() / 2., p.get_height()), 
                        ha='center', va='baseline', fontsize=12, color='black', xytext=(0, 5),
                        textcoords='offset points')

        plt.xlabel(column)
        plt.ylabel('Count')
        if title:
            plt.title(title)
        plt.xticks(rotation=45)
        plt.show()

    def contingency_table(self, col1: str, col2: str):
        """
        Generates a contingency table between two categorical variables.
        
        Args:
        col1 (str): First categorical variable.
        col2 (str): Second categorical variable.
        
        Returns:
        pd.DataFrame: A contingency table showing counts.
        """
        return pd.crosstab(self.df[col1], self.df[col2])

    def plot_stacked_bar_chart(self, col1: str, col2: str, title: str = None):
        """
        Plots a stacked bar chart for two categorical variables.
        
        Args:
        col1 (str): First categorical variable.
        col2 (str): Second categorical variable.
        title (str): Title of the chart (optional).
        """
        crosstab = self.contingency_table(col1, col2)
        crosstab.plot(kind='bar', stacked=True, figsize=(10, 6))
        plt.ylabel('Count')
        if title:
            plt.title(title)
        plt.xticks(rotation=45)
        plt.show()

    def chi_squared_test(self, col1: str, col2: str):
        """
        Performs a chi-squared test to check the independence between two categorical variables.
        
        Args:
        col1 (str): First categorical variable.
        col2 (str): Second categorical variable.
        
        Returns:
        tuple: chi2 statistic, p-value, degrees of freedom, expected frequencies.
        """
        from scipy.stats import chi2_contingency
        contingency = self.contingency_table(col1, col2)
        chi2, p, dof, ex = chi2_contingency(contingency)
        return chi2, p, dof, ex
    
    def test_all_associations(self, categorical_vars: list):
        """
        Tests all pairwise associations between categorical variables using the chi-squared test.
        
        Args:
        categorical_vars (list): List of categorical variables to test.
        
        Returns:
        pd.DataFrame: A DataFrame summarizing the chi-squared test results (p-values) for all pairs, 
        including an 'is_independent' column.
        """
        associations = []

        # Iterate over all combinations of two categorical variables
        for col1, col2 in combinations(categorical_vars, 2):
            chi2, p, dof, ex = self.chi_squared_test(col1, col2)
            is_independent = p > 0.05  # If p-value > 0.05, variables are independent
            associations.append({
                'Variable 1': col1,
                'Variable 2': col2,
                'Chi2': chi2,
                'p-value': p,
                'Degrees of Freedom': dof,
                'is_independent': is_independent
            })

        # Convert the list of associations into a DataFrame for easy viewing
        return pd.DataFrame(associations)