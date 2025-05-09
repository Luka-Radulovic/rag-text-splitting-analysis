from deepeval import evaluate
from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric, ContextualRelevancyMetric
from deepeval.test_case import LLMTestCase
from deepeval.models import OllamaModel
from qdrant_client import QdrantClient
import json
from tqdm import tqdm
from collections import defaultdict
from langchain_ollama import OllamaLLM
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.prompts import PromptTemplate
import numpy as np 
import warnings
from llm import CustomLlama3_8B

warnings.filterwarnings("ignore")

prompt_template = PromptTemplate(
    input_variables=["context", "question"],
    template="""
    Use the following context to answer the question as accurately as possible.

    Context:
    {context}

    Question:
    {question}

    Answer:"""
)

llm = OllamaLLM(model="llama3.1:8b")
chain = prompt_template | llm  

llm_evaluator = CustomLlama3_8B()

qdrant = QdrantClient(url='http://localhost:6333', check_compatibility=False)

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAMES = [
    "char_fullstop_1024_256",
    "char_fullstop_2048_512",
    "char_fullstop_512_128",
    "char_newline_1024_256",
    "char_newline_2048_512",
    "char_newline_512_128",
    "nltk_1024_256",
    "nltk_2048_512",
    "nltk_512_128",
    "recursive_2048_512",
    "recursive_512_128",
]

EVALUATIONS = {
    "char_fullstop_1024_256": {},
    "char_fullstop_2048_512": {},
    "char_fullstop_512_128": {},
    "char_newline_1024_256": {},
    "char_newline_2048_512": {},
    "char_newline_512_128": {},
    "nltk_1024_256": {},
    "nltk_2048_512": {},
    "nltk_512_128": {},
    "recursive_2048_512": {},
    "recursive_512_128": {},
}

METRICS = {
    "answer_relevancy" : AnswerRelevancyMetric(model=llm_evaluator),
    "faithfullness" : FaithfulnessMetric(model=llm_evaluator),
    "contextual_relevancy" : ContextualRelevancyMetric(model=llm_evaluator)
}
embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

with open("questions.json", "r", encoding="utf-8") as f:
    questions = json.load(f)["questions"]

question_embeddings = [embeddings.embed_query(question) for question in questions]

questions = questions[:100]
question_embeddings = question_embeddings[:100]


# main loop 


for collection in tqdm(COLLECTION_NAMES):
    print(f"Currently evaluating collection: {collection}")
    metric_scores = defaultdict(list)

    for query, question in zip(question_embeddings, questions):

        search_result = qdrant.search(
            collection_name=collection,
            query_vector=query,
            limit=5  
        )   
        matched_payloads = [hit.payload for hit in search_result]        
        context_texts = [payload["page_content"] for payload in matched_payloads]

        context = "\n\n".join(context_texts)

        answer = chain.invoke(input={
            "question": question,
            "context": context
        })

        test_case = LLMTestCase(
            input=question,
            actual_output=answer, 
            retrieval_context=[context]
        )

        try:
            result = evaluate(
                test_cases=[test_case],
                metrics=list(METRICS.values())
            )
            for metric_data in result.test_results[0].metrics_data:
                metric_name = metric_data.name.lower().replace(" ", "_")
                metric_scores[metric_name].append(metric_data.score)
        except Exception as e:
            print(f"Evaluation failed for a test case in '{collection}': {e}")
            continue


    for metric_name, scores in metric_scores.items():
        avg_score = sum(scores) / len(scores) if scores else 0.0
        EVALUATIONS[collection][metric_name] = avg_score

with open("evaluations.json", "w") as f:
    json.dump(EVALUATIONS, f, indent=2)

