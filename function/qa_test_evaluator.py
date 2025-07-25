import csv
import time
import io
import os
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import FileResponse
from dotenv import load_dotenv
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# === CONFIGURATION CONSTANTS ===
CHROMA_DIR = "chroma_txt_db_csv"  # Path to ChromaDB directory
EMBED_MODEL_NAME = "meta-llama/llama-4-scout-17b-16e-instruct"  # Embedding model name
LLM_MODEL = "llama3-70b-8192"  # Groq LLM model to use
RESULT_CSV = "qa_test_results.csv"  # Output file for results

# === INITIALIZE FASTAPI APP ===
load_dotenv()
app = FastAPI()

# === INITIALIZE EMBEDDING + VECTOR DB + LLM ===
embedding_model = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
vectordb = Chroma(persist_directory=CHROMA_DIR, embedding_function=embedding_model)
llm = ChatGroq(model_name=LLM_MODEL, temperature=0)

# === Sentence Transformer for answer similarity scoring ===
sentence_model = SentenceTransformer(EMBED_MODEL_NAME)

# === QA RESPONSE FORMAT PROMPT ===
prompt_template = PromptTemplate.from_template("""
You are Antos Pinnacles’ (AP) virtual assistant — a professional and helpful AI trained to support clients of a licensed brokerage firm.

Use the context below to answer the customer's question accurately and concisely. If the information cannot be found in the context, politely admit it.

Context:
{context}

Question: {question}

Instructions:
- Be concise, clear, and professional.
- If the answer is not available in the context, reply with: "I'm sorry, I don't have that information right now."

Only provide the final answer.
""")

# === QA CHAIN SETUP WITH VECTOR RETRIEVAL ===
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectordb.as_retriever(search_kwargs={"k": 5}),
    chain_type_kwargs={"prompt": prompt_template}
)

# === FUNCTION TO COMPARE LLM RESPONSE VS. EXPECTED ANSWER ===
def compute_similarity(ans1, ans2):
    embeddings = sentence_model.encode([ans1, ans2])
    score = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    return round(score, 4)

# === ENDPOINT: Evaluate QA Results from Uploaded CSV ===
@app.post("/evaluate-qa/")
async def evaluate_qa(file: UploadFile = File(...)):
    content = await file.read()
    csv_file = io.StringIO(content.decode("utf-8"))
    reader = csv.DictReader(csv_file)

    results = []

    # Loop through each row of CSV: question → expected answer
    for row in reader:
        question = row['question']
        expected_answer = row['answer']

        # Measure LLM response time
        start_time = time.time()
        response = qa_chain.run(question)
        end_time = time.time()

        # Compute semantic similarity between expected vs. LLM answer
        similarity = compute_similarity(response, expected_answer)
        response_time = round(end_time - start_time, 2)

        # Append results to list
        results.append({
            "Question": question,
            "Expected": expected_answer,
            "LLM_Response": response,
            "Time(s)": response_time,
            "Similarity": similarity
        })

        # Console log for live monitoring
        print(f"✅ Q: {question}")
        print(f"🔁 Response: {response}")
        print(f"⏱️ Time: {response_time}s | 🧠 Similarity: {similarity}")

        # Add a small delay to avoid model/server overload
        time.sleep(0.5)

    # Write results to CSV file
    with open(RESULT_CSV, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)

    # Return the CSV as downloadable response
    return FileResponse(path=RESULT_CSV, filename="qa_test_results.csv", media_type='text/csv')
