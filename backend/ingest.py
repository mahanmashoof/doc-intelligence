import os
import fitz  # PyMuPDF — 'fitz' is its legacy name
from dotenv import load_dotenv
from openai import OpenAI
from pinecone import Pinecone

load_dotenv()

openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(os.environ["PINECONE_INDEX"])

def extract_text(pdf_path: str) -> str:
    doc = fitz.open(pdf_path)
    text = "\n".join([page.get_text() for page in doc])
    doc.close()
    return text

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list[str]:
    words = text.split()
    chunks = []
    start = 0

    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap

    return chunks

def embed_texts(texts: list[str]) -> list[list[float]]:
    response = openai_client.embeddings.create(
        input=texts,
        model="text-embedding-3-small"
    )
    return [item.embedding for item in response.data]

def ingest_pdf(pdf_path: str, doc_id: str) -> int:
    """
    Full pipeline: PDF → chunks → embeddings → Pinecone.
    """
    print(f"Extracting text from {pdf_path}...")
    text = extract_text(pdf_path)

    print("Chunking text...")
    chunks = chunk_text(text)
    print(f"  Created {len(chunks)} chunks")

    print("Generating embeddings...")
    embeddings = embed_texts(chunks)

    print("Uploading to Pinecone...")
    vectors = [
        {
            "id": f"{doc_id}_chunk_{i}",
            "values": embeddings[i],
            "metadata": {
                "text": chunks[i],
                "doc_id": doc_id,
                "chunk_index": i
            }
        }
        for i in range(len(chunks))
    ]

    # Upsert in batches of 100 (Pinecone's recommended batch size)
    batch_size = 100
    for i in range(0, len(vectors), batch_size):
        batch = vectors[i:i + batch_size]
        index.upsert(vectors=batch)

    print(f"Done. Ingested {len(chunks)} chunks.")
    return len(chunks)