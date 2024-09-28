from ragas.integrations.llama_index import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.llms.openai import OpenAI
import pandas as pd

class RAGASEvaluator:
    def __init__(self, evaluator_llm, metrics=None):
        """
        Initialize the RAGAS Evaluator.

        :param evaluator_llm: The LLM used for evaluation, e.g., OpenAI LLM.
        :param metrics: A list of metric functions to use for evaluation.
        """
        self.evaluator_llm = evaluator_llm
        self.metrics = metrics or [faithfulness, answer_relevancy]

    def evaluate(self, query_engine, dataset):
        """
        Evaluate the query engine using the specified metrics.

        :param query_engine: The query engine to be evaluated.
        :param dataset: The dataset containing the questions for evaluation.
        :return: A DataFrame containing the evaluation results.
        """
        result = evaluate(
            query_engine=query_engine,
            metrics=self.metrics,
            dataset=dataset,
            llm=self.evaluator_llm,
            embeddings=OpenAIEmbedding(),
        )
        return result.to_pandas()

    def save_results(self, df_result, path):
        """
        Save the evaluation results to a specified path.

        :param df_result: DataFrame containing the evaluation results.
        :param path: File path where the results should be saved.
        """
        df_result.to_parquet(path)
        print(f"Results saved to {path}")
        print(df_result.describe())

