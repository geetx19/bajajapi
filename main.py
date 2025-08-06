import os
import asyncio
import httpx
from fastapi import FastAPI, HTTPException
from typing import List, Dict
from dotenv import load_dotenv
import google.generativeai as genai
from process_and_store_pdf import query_pinecone
from process_and_store_pdf import store_pdf_in_pinecone
from download_pdf import download_pdf_file
load_dotenv()
from pydantic import BaseModel
from typing import List
import re
class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
INPUT_FOLDER = os.getenv("INPUT_FOLDER","./input")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY not set in .env file.")
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not set in .env file.")

app = FastAPI()
genai.configure(api_key=GOOGLE_API_KEY)


# Stub: Replace with real documents if needed
PROCESSED_DOCS: List[Dict] = [{
    "name": "SampleDoc",
    "summary": "This is a placeholder summary for the purpose of AI querying."
}]

# -------------------------- Query Handler --------------------------

async def query_fallback_ai(questions: List[str], documents: List[Dict]) -> str:
    doc = ""
    for question in questions:
        doc = doc + "".join(query_pinecone(question))
    context =  f"Relevant Context, : {doc}\n" 
    prompt = (
    "You are a helpful AI assistant trained on the following policy documents.\n\n"
    f"User Question: \"{"".join(questions)}\"\n\n"
    f"Relevant Context:\n{context}\n\n"
    "Please answer the user's question **in one clear, complete, and concise sentence**, using the policy context provided. "
    "Include relevant statistics from the documents along with numerical figures wherever possible. "
    "Give the answer in a precise one-liner and include the statistics or numerical measures mentioned in the document.\n\n"
    "📌 Example Answer Format:\n"
    "\"What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?\"\n"
    "➡️ A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.\n\n"
    "\"What is the waiting period for pre-existing diseases (PED) to be covered?\"\n"
    "➡️ There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.\n\n"
    "\"Does the policy cover maternity expenses?\"\n"
    "➡️ Yes, the policy covers maternity expenses including childbirth and lawful medical termination of pregnancy, with a waiting period of 24 months and a limit of two claims.\n\n"
    "If the answer is not found in the context, respond with: 'Information not available in the provided documents.'"
)

        # Fallback: Groq API
    try:
        groq_headers = {
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        }

        body = {
            "model": "meta-llama/llama-4-scout-17b-16e-instruct",  # Or "llama3-70b-8192" or another Groq-supported model
            "messages": [
                {"role": "system", "content": "You are a helpful document assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 800
        }

        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers=groq_headers,
                json=body
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"].strip()

    except httpx.HTTPStatusError as e:
        print(f"Groq HTTP error {e.response.status_code}: {e.response.text}")
    except httpx.RequestError as e:
        print(f"Groq request error: {e}")
    except Exception as e:
        print(f"Groq unexpected error: {e}")

# -------------------------- FastAPI Endpoints --------------------------

@app.get("/documents")
async def get_documents():
    return PROCESSED_DOCS

@app.post("/chat/")
async def chat(question: str):
    if not question:
        raise HTTPException(status_code=400, detail="Empty question.")
    answer = await query_fallback_ai(question, PROCESSED_DOCS)
    return {"answer": answer}

@app.post("/hackrx/run")
async def hackrx_run(request: HackRxRequest):
    
    file_path = download_pdf_file(request.documents, "input.pdf")
    folder =  [os.path.join(INPUT_FOLDER,f) for f in os.listdir(INPUT_FOLDER) if os.path.isfile(os.path.join(INPUT_FOLDER,f))]
    store_pdf_in_pinecone(folder[0])
    results = []
    
    answer = await query_fallback_ai(request.questions,PROCESSED_DOCS)
    results.append(answer)
    chunks = re.split(r'\n?\d+\.\s.*?\n', results[0])

# Step 2: Remove any empty chunks and leading newline artifacts
    answers = [chunk.strip().replace('\n-', ' ') for chunk in chunks if chunk.strip()]

    answers.pop(0)
    return {"answers": answers}
# -------------------------- Terminal Mode --------------------------
@app.get("/")
def read_root():
    return {"status": "API is running"}

def ask_terminal():
    print("Ask anything about your documents (type 'exit' to quit):")
    while True:
        q = input("Your query: ").strip()
        if q.lower() in ['exit', 'quit']:
            break
        if q:
            answer = asyncio.run(query_fallback_ai(q, PROCESSED_DOCS))
            print(f"\n🧠 Answer:\n{answer}\n")

if __name__ == "__main__":
    import uvicorn
    import threading

    def run_server():
        uvicorn.run(app, host="0.0.0.0", port=8000)

    threading.Thread(target=run_server, daemon=True).start()
    ask_terminal()
