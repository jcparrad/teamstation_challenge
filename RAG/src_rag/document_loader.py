import pandas as pd
from llama_index.core import Document

class DocumentLoader:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_documents(self):
        # Load the Excel file
        df = pd.read_excel(self.file_path)

        # Check if the 'text' column exists
        if 'text' not in df.columns:
            raise ValueError("The 'text' column is not present in the provided Excel file.")

        # Create Document objects from the 'text' column
        documents = [Document(text=row['text']) for _, row in df.iterrows()]
        return documents