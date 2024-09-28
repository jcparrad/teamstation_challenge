# src_rag/retriever.py
from llama_index.core.retrievers import VectorIndexRetriever
from llama_index.core import QueryBundle

class Retriever:
    def __init__(self, index):
        self.index = index

    def retrieve_nodes(self, query: str, vector_top_k: int):
        """Retrieves nodes based on the query from the index."""
        query_bundle = QueryBundle(query)  # Creating QueryBundle here
        retriever = VectorIndexRetriever(index=self.index, similarity_top_k=vector_top_k)
        retrieved_nodes = retriever.retrieve(query_bundle)
        return retrieved_nodes, query_bundle  # Return both nodes and query_bundle
