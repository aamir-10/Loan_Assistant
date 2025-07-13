import pandas as pd
import numpy as np
import faiss
import pickle
import os
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load Embedding Model

embedding_model = SentenceTransformer('all-MiniLM-L6-v2')

# Build Knowledge Base from File

def build_knowledge_base():
    with open("data/loan_insights.txt", "r", encoding="utf-8") as f:
        docs = [line.strip() for line in f.readlines() if line.strip()]
    return docs

# Create and Save FAISS Index

def create_faiss_index(docs, index_path="embeddings/vector.index", doc_path="embeddings/docs.pkl"):
    embeddings = embedding_model.encode(docs)

    index = faiss.IndexFlatL2(embeddings.shape[1])
    index.add(np.array(embeddings))

    faiss.write_index(index, index_path)

    with open(doc_path, "wb") as f:
        pickle.dump(docs, f)

    print("✅ FAISS index and docs saved.")

# Load LLM for Answer Generation

llm_model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(llm_model_name)
llm_model = AutoModelForSeq2SeqLM.from_pretrained(llm_model_name)

# Retrieve Top-K Relevant Chunks

def get_relevant_chunks(query, k=5):
    if not os.path.exists("embeddings/vector.index") or not os.path.exists("embeddings/docs.pkl"):
        raise FileNotFoundError("❌ FAISS index or document file not found. Please build the index first.")

    index = faiss.read_index("embeddings/vector.index")
    with open("embeddings/docs.pkl", "rb") as f:
        docs = pickle.load(f)

    query_embedding = embedding_model.encode([query])
    D, I = index.search(np.array(query_embedding), k)

    return [docs[i] for i in I[0]]

# Answer with Fixed Prompt

def generate_answer(query):
    chunks = get_relevant_chunks(query)
    context = "\n".join(chunks)

    prompt = f"""
You are a helpful AI assistant trained to answer questions about loan approvals based on previous bank data.

Context:
{context}

Question:
{query}

Answer with a clear, full-sentence explanation. Don't just give a single word. Use reasoning. For example:
If the user asks "Why was my loan rejected?" and the context shows 'poor credit history', answer like:
"Loans are often rejected when the applicant has a poor credit history, even if other factors like income are sufficient."

Now answer:
"""

    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = llm_model.generate(**inputs, max_new_tokens=150)
    answer = tokenizer.decode(outputs[0], skip_special_tokens=True)

    return answer

# Answer with Custom Prompt

def generate_answer_with_prompt(prompt):
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True)
    outputs = llm_model.generate(**inputs, max_new_tokens=250)
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Optional: CLI to Build Index

if __name__ == "__main__":
    docs = build_knowledge_base()
    create_faiss_index(docs)
