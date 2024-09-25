import os
import sys
from dotenv import load_dotenv

# Dynamically add the EDA/src_eda directory to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
eda_src_path = os.path.join(current_dir, '..', '..', 'EDA', 'src_eda')
sys.path.append(eda_src_path)

from embedding_processor import OpenAIEmbedding  # Now the import should work

# Load environment variables from .env file
load_dotenv()

class TextDataHandler:
    def __init__(self, documents):
        self.documents = documents
        self.embedding_model = OpenAIEmbedding()  # Use the OpenAIEmbedding class

    def generate_embeddings(self):
        """
        Generate embeddings for each document and store them in the 'embedding' attribute of the document.
        
        Returns:
        List[Document]: The list of documents with their embeddings.
        """
        for doc in self.documents:
            doc.embedding = self.embedding_model.get_embedding(doc.text)  # Use the embedding model
        
        return self.documents  # Return the documents with embeddings
