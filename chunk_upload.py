import os
from pathlib import Path
from typing import List
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    CharacterTextSplitter,
    NLTKTextSplitter
)
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Qdrant
from qdrant_client import QdrantClient
from tqdm import tqdm

PDF_FOLDER = Path("pdfs")
CHUNK_STRATEGIES = {
    "recursive": RecursiveCharacterTextSplitter,
    "char": CharacterTextSplitter,
    "nltk": NLTKTextSplitter,
}
CHUNK_CONFIGS = [
    {"chunk_size": 512, "chunk_overlap": 128},
    {"chunk_size": 1024, "chunk_overlap": 256},
    {"chunk_size": 2048, "chunk_overlap": 512},
]

EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
QDRANT_HOST = "localhost"
QDRANT_PORT = 6333


def load_pdfs(pdf_folder: Path) -> List[str]:
    documents = []
    for file in tqdm(pdf_folder.glob("*.pdf"), desc="Loading PDFs"):
        loader = PyPDFLoader(str(file))
        docs = loader.load()
        documents.extend(docs)
    return documents


def get_splitter(strategy_name: str, chunk_size: int, chunk_overlap: int):
    splitter_cls = CHUNK_STRATEGIES[strategy_name]
    if strategy_name == "nltk":
        return splitter_cls(chunk_size=chunk_size)
    return splitter_cls(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


def create_and_upload_collection(strategy: str, chunk_cfg: dict, documents: List[str], embeddings):
    chunk_size = chunk_cfg["chunk_size"]
    chunk_overlap = chunk_cfg["chunk_overlap"]
    collection_name = f"{strategy}_{chunk_size}_{chunk_overlap}"

    print(f"\n Processing with {collection_name}...")

    splitter = get_splitter(strategy, chunk_size, chunk_overlap)
    split_docs = splitter.split_documents(documents)

    qdrant = Qdrant.from_documents(
        documents=split_docs,
        embedding=embeddings,
        collection_name=collection_name,
        location=QDRANT_HOST,
        port=QDRANT_PORT,
    )

    print(f"Uploaded {len(split_docs)} chunks to Qdrant collection: {collection_name}")


def main():
    documents = load_pdfs(PDF_FOLDER)
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    for strategy in CHUNK_STRATEGIES:
        for cfg in CHUNK_CONFIGS:
            create_and_upload_collection(strategy, cfg, documents, embeddings)


if __name__ == "__main__":
    main()
