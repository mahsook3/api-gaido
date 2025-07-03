import os
import requests
from pypdf import PdfReader
import re
from dotenv import load_dotenv

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate

# -----------------------------
# Load environment variables
# -----------------------------
load_dotenv()

gemini_api_key = os.getenv("GEMINI_API_KEY")
if not gemini_api_key:
    raise ValueError("Gemini API Key not found. Please set GEMINI_API_KEY in your .env file")

# -----------------------------
# Download PDF
# -----------------------------
def download_pdf(url, save_path):
    if os.path.exists(save_path):
        print(f"PDF already exists at {save_path}")
        return
    
    response = requests.get(url)
    response.raise_for_status()
    with open(save_path, "wb") as f:
        f.write(response.content)
    print(f"PDF downloaded to {save_path}")

pdf_url = "https://all.docs.genesys.com/images/pdf/en-GWE-8.1.2-API-book.pdf"
pdf_path = "en-GWE-8.1.2-API-book.pdf"

download_pdf(pdf_url, pdf_path)

# -----------------------------
# Load and extract PDF text with better preprocessing
# -----------------------------
def load_pdf_text(file_path):
    reader = PdfReader(file_path)
    text = ""
    for page_num, page in enumerate(reader.pages):
        try:
            page_text = page.extract_text()
            if page_text:
                # Clean up the text
                page_text = re.sub(r'\s+', ' ', page_text)  # Replace multiple spaces with single space
                page_text = re.sub(r'\n+', '\n', page_text)  # Replace multiple newlines with single newline
                text += f"Page {page_num + 1}: {page_text}\n\n"
        except Exception as e:
            print(f"Error extracting text from page {page_num + 1}: {e}")
    return text

pdf_text = load_pdf_text(pdf_path)
print(f"Extracted text length: {len(pdf_text)} characters")

# -----------------------------
# Split text into chunks with better parameters
# -----------------------------
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,  # Increased chunk size for better context
    chunk_overlap=200,  # Increased overlap for better continuity
    separators=["\n\n", "\n", ". ", "? ", "! ", " ", ""],
    length_function=len,
)

chunks = text_splitter.split_text(pdf_text)
print(f"Number of text chunks: {len(chunks)}")

# Filter out very short chunks that might not be useful
chunks = [chunk for chunk in chunks if len(chunk.strip()) > 100]
print(f"Number of chunks after filtering: {len(chunks)}")

# -----------------------------
# Create Gemini embedding model
# -----------------------------
embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",
    google_api_key=gemini_api_key
)

# -----------------------------
# Create Chroma vectorstore with better configuration
# -----------------------------
persist_dir = "chroma_db"

# Check if vectorstore already exists
if os.path.exists(persist_dir):
    print("Loading existing vectorstore...")
    vectorstore = Chroma(
        persist_directory=persist_dir,
        embedding_function=embedding_model
    )
else:
    print("Creating new vectorstore...")
    vectorstore = Chroma.from_texts(
        texts=chunks,
        embedding=embedding_model,
        persist_directory=persist_dir,
        metadatas=[{"chunk_id": i} for i in range(len(chunks))]
    )

print("Vectorstore ready.")

# -----------------------------
# Define Gemini chat model with better configuration
# -----------------------------
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=gemini_api_key,
    temperature=0.1,  # Lower temperature for more focused responses
    max_output_tokens=1000
)

# -----------------------------
# Define improved RAG prompt
# -----------------------------
template = """You are an expert assistant helping users understand technical documentation. 
Use the provided context to answer the question accurately and comprehensively.

Instructions:
1. Base your answer primarily on the provided context
2. If the context doesn't contain enough information, clearly state what information is missing
3. Provide specific details and examples when available
4. Use clear, professional language
5. If the question is about a specific feature or process, explain it step by step
6. you are also capable of providing code examples if relevant based on the context
7. Provide the response only in Markdown format, without any additional text or formatting
8. Reponse in MD code
Context:
{context}

Question: {question}

Answer:"""

prompt = PromptTemplate(
    template=template,
    input_variables=["context", "question"]
)

# -----------------------------
# Create RAG chain with better retrieval
# -----------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 5}  # Retrieve more relevant chunks
    ),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt},
    return_source_documents=True
)

# -----------------------------
# Enhanced query processing
# -----------------------------
def preprocess_query(query):
    """Preprocess the query to improve retrieval"""
    # Remove extra whitespace
    query = re.sub(r'\s+', ' ', query.strip())
    return query

def post_process_answer(answer):
    """Post-process the answer to improve readability"""
    # Remove extra whitespace
    answer = re.sub(r'\s+', ' ', answer.strip())
    return answer

# -----------------------------
# Interactive loop with better error handling
# -----------------------------
def interactive_loop():
    print("\n" + "="*50)
    print("Welcome to the Genesys API Documentation Assistant!")
    print("Ask questions about the API documentation.")
    print("Type 'quit' to exit, 'help' for tips.")
    print("="*50)
    
    while True:
        try:
            user_input = input("\nEnter your question: ").strip()
            
            if user_input.lower() == "quit":
                print("Goodbye!")
                break
            elif user_input.lower() == "help":
                print("\nTips for better results:")
                print("- Ask specific questions about API features")
                print("- Use clear, descriptive language")
                print("- Ask about specific functions, endpoints, or processes")
                print("- Example: 'How do I authenticate with the API?'")
                continue
            elif not user_input:
                print("Please enter a question.")
                continue

            # Preprocess the query
            processed_query = preprocess_query(user_input)
            
            # Get the answer
            result = qa_chain.invoke(processed_query)
            
            # Post-process and display the answer
            answer = post_process_answer(result["result"])
            
            print("\n" + "="*20 + " ANSWER " + "="*20)
            print(answer)
            
            # Optional: Show source information
            if result.get("source_documents"):
                print(f"\n[Based on {len(result['source_documents'])} relevant sections from the documentation]")
                
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"An error occurred: {e}")
            print("Please try again with a different question.")

# -----------------------------
# Test function to verify the system
# -----------------------------
def test_system():
    """Test the RAG system with sample queries"""
    test_queries = [
        "What is this document about?",
        "What are the main API features?",
        "How do I authenticate?",
        "What endpoints are available?"
    ]
    
    print("\n" + "="*30)
    print("Testing the RAG system...")
    print("="*30)
    
    for query in test_queries:
        print(f"\nTest Query: {query}")
        try:
            result = qa_chain.invoke(query)
            print(f"Answer: {result['result'][:200]}...")
        except Exception as e:
            print(f"Error: {e}")

if __name__ == "__main__":
    # Uncomment the line below to run tests first
    # test_system()
    
    interactive_loop()