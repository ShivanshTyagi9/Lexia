from datasets import Dataset
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness,
)
from ragas import evaluate
import pandas as pd
import requests

# --- CONFIG ---
API_ENDPOINT = "http://localhost:8000/query"  # Change if your FastAPI runs on a different port

# --- Step 1: Define your test dataset (manual or CSV import) ---
# You can also load from a CSV file with columns: question, ground_truth

test_data = [
    {
        "question": "What is the main idea of the CoALA paper?",
        "ground_truth": "The CoALA paper introduces a method to align language agents with human values through cooperative AI techniques."
    },
    {
        "question": "How does RAG differ from traditional search?",
        "ground_truth": "RAG combines vector retrieval with generative models to provide context-aware answers, unlike traditional keyword-based search."
    },
    {
        "question": "What embedding model is used in this system?",
        "ground_truth": "The system uses the nomic-embed-text model for document embeddings."
    },
]

# --- Step 2: Get model-generated answers from your RAG API ---
def get_rag_answer_and_context(question):
    try:
        response = requests.get(API_ENDPOINT, params={"question": question})
        if response.status_code == 200:
            json = response.json()
            return json.get("answer", ""), json.get("contexts", [])
        else:
            return f"[ERROR {response.status_code}]", []
    except Exception as e:
        return str(e), []

records = []
for row in test_data:
    question = row["question"]
    gt = row["ground_truth"]
    answer, contexts = get_rag_answer_and_context(question)

    records.append({
        "question": question,
        "ground_truth": gt,
        "answer": answer,
        "contexts": contexts,
    })

dataset = Dataset.from_pandas(pd.DataFrame(records))

results = evaluate(
    dataset,
    metrics=[
        answer_relevancy,
        faithfulness,
        context_precision,
        context_recall,
    ]
)

print("\n📊 RAGAS Evaluation Results:")
print(results.to_pandas().mean())
