# src_rag/response_synthesizer.py
from llama_index.core import get_response_synthesizer

class ResponseSynthesizer:
    def __init__(self):
        self.synthesizer = get_response_synthesizer(response_mode="compact")

    def synthesize_response(self, query: str, nodes):
        """Synthesizes a response from the retrieved nodes."""
        response = self.synthesizer.synthesize(query, nodes=nodes)
        return response
