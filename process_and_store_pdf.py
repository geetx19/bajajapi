import os
import uuid
import pdfplumber
from pinecone import Pinecone
from dotenv import load_dotenv

load_dotenv()

# Setup Pinecone client
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))

# Create the index if it doesn't exist
index_name = "demo"

if not pc.has_index(index_name):
    pc.create_index_for_model(
        name=index_name,
        cloud="aws",  # or "gcp" depending on your account
        region="us-east-1",  # adapt if needed
        embed={
            "model": "llama-text-embed-v2",  # Pinecone native embedding
            "field_map": {"text": "values"}
        }
    )

index = pc.Index(index_name)

import fitz  # PyMuPDF

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with fitz.open(pdf_path) as doc:
        for page in doc:
            text += page.get_text() + "\n"
    return text

# Chunking helper
def chunk_text(text: str, chunk_size: int = 500) -> list:
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chunks.append(chunk)
    return chunks

# Store chunks in Pinecone
def store_pdf_in_pinecone(pdf_path: str, metadata: dict = None):
    text = extract_text_from_pdf(pdf_path)
    if not text:
        print(f"No extractable text in {pdf_path}")
        return
    chunks = chunk_text(text)
    print(chunks)

    embeddings = pc.inference.embed(
    model="llama-text-embed-v2",
    inputs=[chunk for chunk in chunks],
    parameters={"input_type": "passage", "truncate": "END"})
    docs_to_upsert = []
    for i,(chunk, vector) in enumerate(zip(chunks,(embeddings.data))):
        doc = {
            "id": f"{uuid.uuid4()}",
            "values":vector["values"],
            "metadata":{  "source": os.path.basename(pdf_path),"chunk": i,"text":chunk}
        }
        docs_to_upsert.append(doc)
    index.upsert(docs_to_upsert)
    print(docs_to_upsert)
    print(f"Uploaded {len(docs_to_upsert)} chunks from {pdf_path} to Pinecone.")

def query_pinecone(query_text: str, top_k: int = 5):
    # Step 1: Embed the query
    query_embedding = pc.inference.embed(
        model="llama-text-embed-v2",
        inputs=[query_text],
        parameters={"input_type": "query", "truncate": "END"}
    ).data[0]["values"]  # Single query -> take first embedding

    # Step 2: Query the Pinecone index
    result = index.query(
        vector=query_embedding,
        top_k=1,
        include_metadata=True  # So we get the chunk text & source info
    )
    doc = []
    for match in result["matches"]:
        doc.append(match["metadata"].get("text"))
    return doc
if __name__ == "__main__":
    sample_pdf = "./input/example_policy.pdf"
    INPUT_FOLDER = os.getenv("INPUT_FOLDER")
    folder =  [os.path.join(INPUT_FOLDER,f) for f in os.listdir(INPUT_FOLDER) if os.path.isfile(os.path.join(INPUT_FOLDER,f))]
    #store_pdf_in_pinecone(folder[0])
    query_pinecone(" If insured has travelled a distance of 300kms using an air ambulance we will pay 50% of the total cost or Sum Insured whichever is lower. (Eligibility/Actual distance travelled : 150kms/300kms = 0.5) ")

