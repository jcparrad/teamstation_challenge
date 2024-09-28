# src_rag/rag_query_engine.py
class RAGQueryEngine:
    def __init__(self, retriever, reranker, synthesizer, vector_top_k=10, reranker_top_n=3, with_reranker=False):
        """
        Initializes the RAGQueryEngine with the necessary components and query settings.
        
        :param retriever: The retriever instance used for node retrieval.
        :param reranker: The reranker instance used for reranking nodes.
        :param synthesizer: The response synthesizer instance used for response generation.
        :param vector_top_k: Number of top results to retrieve.
        :param reranker_top_n: Number of top results to rerank.
        :param with_reranker: Flag to indicate if reranking should be applied.
        """
        self.retriever = retriever
        self.reranker = reranker
        self.synthesizer = synthesizer
        self.vector_top_k = vector_top_k
        self.reranker_top_n = reranker_top_n
        self.with_reranker = with_reranker

    def query(self, query_str: str):
        """
        Handles the overall query flow: retrieval, optional reranking, and response synthesis.
        
        :param query_str: The query string input by the user.
        :return: Synthesized response based on retrieved and optionally reranked nodes.
        """
        retrieved_nodes, query_bundle = self.retriever.retrieve_nodes(query_str, self.vector_top_k)
        if self.with_reranker:
            retrieved_nodes = self.reranker.rerank_nodes(retrieved_nodes, query_bundle, self.reranker_top_n)
        response = self.synthesizer.synthesize_response(query_str, retrieved_nodes)
        return response
