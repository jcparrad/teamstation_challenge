import pandas as pd

class DateTimeProcessor:
    def __init__(self, df: pd.DataFrame, open_col: str, closed_col: str):
        """
        Initializes the DateTimeProcessor class with a dataframe and datetime columns.
        
        Args:
        df (pd.DataFrame): The dataframe containing datetime columns.
        open_col (str): The column name representing the 'Date_Time_Open'.
        closed_col (str): The column name representing the 'Date_Time_Closed'.
        """
        self.df = df
        self.open_col = open_col
        self.closed_col = closed_col
    def calculate_duration(self) -> pd.DataFrame:
        """
        Adds a new column to the dataframe with the duration between open and closed times in minutes.
        
        Returns:
        pd.DataFrame: The dataframe with an additional 'Duration' column (in minutes).
        """
        self.df['Duration'] = (self.df[self.closed_col] - self.df[self.open_col]).dt.total_seconds() / 60  # Convert to minutes
        return self.df

    def extract_datetime_features(self) -> pd.DataFrame:
        """
        Adds new columns to the dataframe for day of the week, week number, and month.
        
        Returns:
        pd.DataFrame: The dataframe with additional columns for day of the week, week number, and month.
        """
        self.df['Day_of_Week'] = self.df[self.open_col].dt.day_name()
        self.df['Week_Number'] = self.df[self.open_col].dt.isocalendar().week
        self.df['Month'] = self.df[self.open_col].dt.month_name()
        self.df['Year'] = self.df[self.open_col].dt.year
        return self.df

    def process_all(self) -> pd.DataFrame:
        """
        Process the dataframe to add all calculated features.
        
        Returns:
        pd.DataFrame: The dataframe with 'Duration', 'Day_of_Week', 'Week_Number', and 'Month' columns.
        """
        self.calculate_duration()
        self.extract_datetime_features()
        return self.df
