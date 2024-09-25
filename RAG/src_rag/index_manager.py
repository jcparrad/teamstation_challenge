from llama_index.core import VectorStoreIndex

class IndexManager:
    def __init__(self):
        self.index = None

    def create_index(self, documents):
        self.index = VectorStoreIndex.from_documents(documents)

    def query_index(self, query):
        if self.index is None:
            raise ValueError("Index has not been created yet.")
        query_engine = self.index.as_query_engine()
        return query_engine.query(query)