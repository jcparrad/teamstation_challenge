import os
import openai
import pandas as pd
from dotenv import load_dotenv
from typing import List

from typing import List, Union


class OpenAIEmbedding:
    def __init__(self):
        """
        Initializes the OpenAIEmbedding class and loads the API key from the .env file.
        """
        load_dotenv()  # Load environment variables from .env file
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("API Key not found in .env file.")
        
        openai.api_key = self.api_key

    def get_embedding(self, text: Union[str, None]) -> Union[List[float], None]:
        """
        Retrieves the embedding for a given text using the OpenAI API.
        
        Args:
        text (str): The input text to embed.

        Returns:
        List[float] or None: The embedding vector for the input text, or None if input is invalid.
        """
        if not isinstance(text, str):
            print(f"Invalid input for embedding: {text}. Expected a string.")
            return None
        
        try:
            response = openai.embeddings.create(
                input=[text],  # Wrap the text in a list to send it as a valid JSON
                model="text-embedding-ada-002"  # You can change the model if needed
            )
            # Access the embedding from the response object
            return response.data[0].embedding
        except Exception as e:
            print(f"Error while fetching embedding: {e}")
            return None



class EmbeddingProcessor:
    def __init__(self, df: pd.DataFrame, text_columns: List[str], embedding_model: OpenAIEmbedding):
        """
        Initializes the EmbeddingProcessor class.
        
        Args:
        df (pd.DataFrame): The dataframe containing text data.
        text_columns (List[str]): List of columns containing text to embed.
        embedding_model (OpenAIEmbedding): An instance of the OpenAIEmbedding class.
        """
        self.df = df
        self.text_columns = text_columns
        self.embedding_model = embedding_model

    def generate_embeddings(self) -> pd.DataFrame:
        """
        Adds embedding columns to the dataframe for each specified text column.

        Returns:
        pd.DataFrame: The dataframe with added embedding columns.
        """
        for column in self.text_columns:
            embeddings = self.df[column].apply(self.embedding_model.get_embedding)
            self.df[f"{column}_embedding"] = embeddings
        return self.df
