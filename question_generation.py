import json
from pathlib import Path
from langchain_ollama import ChatOllama
from langchain_qdrant import Qdrant
from langchain_core.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, SystemMessagePromptTemplate
from langchain.chains import LLMChain
from qdrant_client import QdrantClient
from tqdm import tqdm
import re 


COLLECTION_NAME = "recursive_1024_256"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
OLLAMA_MODEL = "llama3:8b"
OUTPUT_JSON = "generated_questions.json"
NUM_QUESTIONS = 500

from langchain_community.embeddings import HuggingFaceEmbeddings

qdrant_client = QdrantClient(
    host=QDRANT_HOST,
    port=QDRANT_PORT,
)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

qdrant = Qdrant(
    client=qdrant_client,
    collection_name=COLLECTION_NAME,
    embeddings=embeddings
)
print("Fetching documents from Qdrant...")
documents = qdrant.similarity_search("Alzheimer's disease", k=NUM_QUESTIONS)  # Fetch extra
texts = [doc.page_content for doc in documents]

with open(file='question_generation_system_prompt.txt', mode ='r') as file:
            prompt_content = file.read()
            system_prompt = prompt_content

with open(file='question_generation_human_prompt.txt', mode ='r') as file:
            prompt_content = file.read()
            human_prompt = prompt_content


question_generation_prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    system_prompt,
                ),
                (
                    "human",
                    human_prompt
                ),
            ]
        )

llm = ChatOllama(
    base_url="http://localhost:11434",  
    model=OLLAMA_MODEL,
    temperature=0.7,
)

qg_chain = question_generation_prompt | llm 

questions = [] 


for input_chunk in tqdm(texts):
    res = qg_chain.invoke(input={"input_chunk":input_chunk})
    raw_response = res.content 
    match = re.search(r'<(.*?)>', raw_response)
    if match:
        question = match.group(1)
        questions.append(question)
    else:
        print("No question found.")
    


with open("questions.json", "w", encoding="utf-8") as f:
    json.dump({"questions": questions}, f, ensure_ascii=False, indent=2)

