import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

class MissingDataAnalyzer:
    def __init__(self, dataframe):
        self.dataframe = dataframe

    def missing_data_summary(self):
        """Generates a summary of missing values in the DataFrame, including counts and percentages."""
        missing_count = self.dataframe.isnull().sum()  # Count of missing values
        missing_percentage = (missing_count / len(self.dataframe)) * 100  # Percentage of missing values
        missing_info = pd.DataFrame({'Missing Count': missing_count, 'Missing Percentage': missing_percentage})
        missing_info = missing_info[missing_info['Missing Count'] > 0]  # Filter columns with missing values
        return missing_info

    def visualize_missing_data(self):
        """Visualizes missing data using a heatmap."""
        plt.figure(figsize=(12, 6))
        sns.heatmap(self.dataframe.isnull(), cbar=False, cmap='viridis')
        plt.title('Missing Data Heatmap')
        plt.xlabel('Columns')
        plt.ylabel('Rows')
        plt.show()

    def analyze_missing_data(self):
        """Combines the summary and visualization of missing data."""
        summary = self.missing_data_summary()
        print("Missing Data Summary:\n", summary)
        self.visualize_missing_data()
