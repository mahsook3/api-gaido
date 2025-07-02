import requests
from pypdf import PdfReader
import os
import re
import google.generativeai as genai
from chromadb import Documents, EmbeddingFunction, Embeddings
import chromadb
from typing import List
from dotenv import load_dotenv

# Disable ChromaDB telemetry completely to avoid error messages
os.environ["ANONYMIZED_TELEMETRY"] = "False"
os.environ["CHROMA_SERVER_NOFILE"] = "1"
os.environ["ALLOW_RESET"] = "TRUE"
# Disable all ChromaDB logging and telemetry
import logging
logging.getLogger("chromadb").setLevel(logging.CRITICAL)
logging.getLogger("chromadb.telemetry").setLevel(logging.CRITICAL)

# Suppress warnings
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

# Load environment variables from .env file
load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

# Download the PDF from the specified URL and save it to the given path
def download_pdf(url, save_path):
    response = requests.get(url)
    with open(save_path, 'wb') as f:
        f.write(response.content)

# URL and local path for the PDF document
pdf_url = "https://all.docs.genesys.com/images/pdf/en-GWE-8.1.2-API-book.pdf"
pdf_path = "en-GWE-8.1.2-API-book.pdf"
download_pdf(pdf_url, pdf_path)

# Load the PDF file and extract text from each page
def load_pdf(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text
    return text

pdf_text = load_pdf(pdf_path)

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("Gemini API Key not provided or incorrect. Please provide a valid GEMINI_API_KEY.")
try:
    genai.configure(api_key=gemini_api_key)
    print("API configured successfully with the provided key.")
except Exception as e:
    print("Failed to configure API:", str(e))

# Split the text into chunks based on double newlines
def split_text(text):
    return [i for i in re.split('\n\n', text) if i.strip()]

chunked_text = split_text(pdf_text)

# Define a custom embedding function using Gemini API
class GeminiEmbeddingFunction(EmbeddingFunction):
    def __init__(self):
        """Initialize the Gemini embedding function."""
        pass
    
    def __call__(self, input: Documents) -> Embeddings:
        gemini_api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=gemini_api_key)
        model = "models/embedding-001"
        title = "Custom query"
        return genai.embed_content(model=model, content=input, task_type="retrieval_document", title=title)["embedding"]

# Create directory for database if it doesn't exist
db_folder = "chroma_db"
if not os.path.exists(db_folder):
    os.makedirs(db_folder)

# Create a Chroma database with the given documents
def create_chroma_db(documents: List[str], path: str, name: str):
    # Suppress stdout/stderr to hide telemetry errors
    import sys
    from contextlib import redirect_stderr, redirect_stdout
    from io import StringIO
    
    try:
        chroma_client = chromadb.PersistentClient(path=path)
        # Try to get existing collection first
        with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
            db = chroma_client.get_collection(name=name, embedding_function=GeminiEmbeddingFunction())
        print(f"Loaded existing collection '{name}' with {db.count()} documents.")
        return db, name
    except:
        # Create new collection if it doesn't exist
        with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
            db = chroma_client.create_collection(name=name, embedding_function=GeminiEmbeddingFunction())
        print(f"Created new collection '{name}'. Adding {len(documents)} documents...")
        for i, d in enumerate(documents):
            with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
                db.add(documents=[d], ids=[str(i)])
        print(f"Added {len(documents)} documents to the collection.")
        return db, name

# Specify the path and collection name for Chroma database
db_name = "rag_experiment"
db_path = os.path.join(os.getcwd(), db_folder)
db, db_name = create_chroma_db(chunked_text, db_path, db_name)

# Retrieve the most relevant passages based on the query
def get_relevant_passage(query: str, db, n_results: int):
    # Suppress stdout/stderr to hide telemetry errors
    from contextlib import redirect_stderr, redirect_stdout
    from io import StringIO
    
    with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
        results = db.query(query_texts=[query], n_results=n_results)
    return [doc[0] for doc in results['documents']]

query = "What is the AI Maturity Scale?"
relevant_text = get_relevant_passage(query, db, n_results=1)

# Construct a prompt for the generation model based on the query and retrieved data
def make_rag_prompt(query: str, relevant_passage: str):
    escaped_passage = relevant_passage.replace("'", "").replace('"', "").replace("\n", " ")
    prompt = f"""You are a helpful and informative bot that answers questions using text from the reference passage included below.
Be sure to respond in a complete sentence, being comprehensive, including all relevant background information.
However, you are talking to a non-technical audience, so be sure to break down complicated concepts and
strike a friendly and conversational tone.
QUESTION: '{query}'
PASSAGE: '{escaped_passage}'

ANSWER:
"""
    return prompt

# Generate an answer using the Gemini Pro API
def generate_answer(prompt: str):
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    result = model.generate_content(prompt)
    return result.text

# Construct the prompt and generate the answer
final_prompt = make_rag_prompt(query, "".join(relevant_text))
answer = generate_answer(final_prompt)
print(answer)

# Interactive function to process user input and generate an answer
def process_query_and_generate_answer():
    while True:
        query = input("\nPlease enter your query (or 'quit' to exit): ")
        if not query or query.lower() == 'quit':
            print("Goodbye!")
            break
        relevant_text = get_relevant_passage(query, db, n_results=1)
        if not relevant_text:
            print("No relevant information found for the given query.")
            continue
        final_prompt = make_rag_prompt(query, "".join(relevant_text))
        answer = generate_answer(final_prompt)
        print("\nGenerated Answer:", answer)

# Main execution
if __name__ == "__main__":
    # Invoke the function to interact with user
    process_query_and_generate_answer()