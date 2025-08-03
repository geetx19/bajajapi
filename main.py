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

class HackRxRequest(BaseModel):
    documents: str
    questions: List[str]

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

async def query_fallback_ai(question: str, documents: List[Dict]) -> str:
    doc = query_pinecone(question)
    context = "\n\n==========\n\n".join(
        f"Relevant Context, Rank {i} : {d}\n" for i,d in enumerate(doc)
    )
    prompt = (
         "You are a helpful AI assistant trained on the following policy documents.\n"
          f"User Question: \"{question}\"\n\n"
          f"Relevant Context:\n{context}\n\n"
          "Please answer the user's question **in one clear, complete, and concise sentence**, using the policy context provided. Include relevant statistics from the documents along with numerical figures wherever possible"
          "If the answer is not found in the context, respond with 'Information not available in the provided documents.'")

    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        print(f"Gemini (Google SDK) failed: {e}")

    # Fallback: OpenRouter
    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "HTTP-Referer": "https://rag-llm-system.onrender.com",
        "X-Title": "DocAnalyzer Assistant"
    }
    try:
        body = {
            "model": "meta-llama/llama-3-8b-instruct",
            "messages": [
                {"role": "system", "content": "You are a helpful document assistant."},
                {"role": "user", "content": prompt}
            ],
            "temperature": 0.3,
            "max_tokens": 800
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post("https://openrouter.ai/api/v1/chat/completions", json=body, headers=headers)
            resp.raise_for_status()
            return resp.json()['choices'][0]['message']['content'].strip()
    except Exception as e:
        print(f"LLaMA via OpenRouter failed: {e}")

    return "Error: All model fallbacks failed."

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
    for question in request.questions:
        answer = await query_fallback_ai(question,PROCESSED_DOCS)
        results.append(answer)

    return {"answers": results}
# -------------------------- Terminal Mode --------------------------

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
