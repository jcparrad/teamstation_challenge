# src_rag/reranker.py
from llama_index.llms.openai import OpenAI
from llama_index.postprocessor.rankgpt_rerank import RankGPTRerank

class Reranker:
    def __init__(self, api_key: str):
        self.api_key = api_key

    def rerank_nodes(self, nodes, query_bundle, top_n):
        """Reranks retrieved nodes based on the query."""
        reranker = RankGPTRerank(
            llm=OpenAI(model="gpt-3.5-turbo", temperature=0.0, api_key=self.api_key),
            top_n=top_n,
            verbose=True,
        )
        reranked_nodes = reranker.postprocess_nodes(nodes, query_bundle)  # Using query_bundle
        return reranked_nodes
