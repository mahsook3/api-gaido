import gradio as gr
import requests
from pypdf import PdfReader
import os
import re
import google.generativeai as genai
from chromadb import Documents, EmbeddingFunction, Embeddings
import chromadb
from typing import List
from dotenv import load_dotenv
import sys
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO

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

# Initialize database connection
def initialize_database():
    """Initialize connection to existing ChromaDB database"""
    db_folder = "chroma_db"
    db_name = "rag_experiment"
    db_path = os.path.join(os.getcwd(), db_folder)
    
    try:
        chroma_client = chromadb.PersistentClient(path=db_path)
        with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
            db = chroma_client.get_collection(name=db_name, embedding_function=GeminiEmbeddingFunction())
        print(f"Connected to existing collection '{db_name}' with {db.count()} documents.")
        return db
    except Exception as e:
        raise Exception(f"Failed to connect to database. Please run app.py first to create the database. Error: {str(e)}")

# Retrieve the most relevant passages based on the query
def get_relevant_passage(query: str, db, n_results: int):
    """Retrieve relevant passages from the database"""
    with redirect_stderr(StringIO()), redirect_stdout(StringIO()):
        results = db.query(query_texts=[query], n_results=n_results)
    return [doc[0] for doc in results['documents']]

# Construct a prompt for the generation model based on the query and retrieved data
def make_rag_prompt(query: str, relevant_passage: str):
    """Create a RAG prompt for answer generation"""
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
    """Generate answer using Gemini API"""
    gemini_api_key = os.getenv("GEMINI_API_KEY")
    if not gemini_api_key:
        raise ValueError("Gemini API Key not provided. Please provide GEMINI_API_KEY as an environment variable")
    genai.configure(api_key=gemini_api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
    result = model.generate_content(prompt)
    return result.text

# Main function to process queries
def process_question(question: str, num_results: int = 3):
    """Process user question and return answer with relevant passages"""
    try:
        if not question.strip():
            return "Please enter a valid question.", ""
        
        # Get relevant passages
        relevant_passages = get_relevant_passage(question, db, n_results=num_results)
        
        if not relevant_passages:
            return "No relevant information found for your question.", ""
        
        # Combine passages
        combined_passage = " ".join(relevant_passages)
        
        # Generate answer
        prompt = make_rag_prompt(question, combined_passage)
        answer = generate_answer(prompt)
        
        # Format relevant passages for display
        formatted_passages = "\n\n".join([f"**Passage {i+1}:**\n{passage}" for i, passage in enumerate(relevant_passages)])
        
        return answer, formatted_passages
        
    except Exception as e:
        return f"Error processing your question: {str(e)}", ""

# Initialize database at startup
try:
    db = initialize_database()
    db_status = f"✅ Database connected successfully! Collection contains {db.count()} documents."
except Exception as e:
    db = None
    db_status = f"❌ Database connection failed: {str(e)}"

# Create Gradio interface
def create_gradio_interface():
    """Create and configure the Gradio interface"""
    
    with gr.Blocks(title="RAG Question Answering System", theme=gr.themes.Soft()) as interface:
        gr.Markdown(
            """
            # 📚 RAG Question Answering System
            
            Ask questions about your documents and get AI-powered answers based on relevant content from your knowledge base.
            
            **How it works:**
            1. Enter your question in the text box below
            2. Adjust the number of relevant passages to retrieve (optional)
            3. Click "Get Answer" to receive an AI-generated response
            4. View the relevant source passages that were used to generate the answer
            """
        )
        
        # Database status
        gr.Markdown(f"**Database Status:** {db_status}")
        
        if db is not None:
            # Main interface
            with gr.Row():
                with gr.Column(scale=2):
                    question_input = gr.Textbox(
                        label="Your Question",
                        placeholder="Enter your question here... (e.g., 'What is the AI Maturity Scale?')",
                        lines=3
                    )
                    
                    num_results = gr.Slider(
                        minimum=1,
                        maximum=5,
                        value=3,
                        step=1,
                        label="Number of relevant passages to retrieve",
                        info="More passages provide more context but may be less focused"
                    )
                    
                    submit_btn = gr.Button("Get Answer", variant="primary", size="lg")
                    
                with gr.Column(scale=1):
                    gr.Markdown(
                        """
                        ### 💡 Tips:
                        - Be specific in your questions
                        - Try different phrasings if you don't get good results
                        - Use keywords related to your documents
                        """
                    )
            
            # Output section
            with gr.Row():
                with gr.Column():
                    answer_output = gr.Textbox(
                        label="AI Generated Answer",
                        lines=8,
                        interactive=False
                    )
                    
                    passages_output = gr.Textbox(
                        label="Relevant Source Passages",
                        lines=10,
                        interactive=False
                    )
            
            # Example questions
            gr.Examples(
                examples=[
                    ["What is the AI Maturity Scale?", 3],
                    ["How does the framework help organizations?", 2],
                    ["What are the key components mentioned?", 3],
                ],
                inputs=[question_input, num_results],
                label="Example Questions"
            )
            
            # Connect the submit button to the processing function
            submit_btn.click(
                fn=process_question,
                inputs=[question_input, num_results],
                outputs=[answer_output, passages_output]
            )
            
            # Also allow Enter key to submit
            question_input.submit(
                fn=process_question,
                inputs=[question_input, num_results],
                outputs=[answer_output, passages_output]
            )
        
        else:
            gr.Markdown(
                """
                ### ⚠️ Database Not Available
                
                Please run `app.py` first to create and populate the database with documents.
                
                ```bash
                python app.py
                ```
                
                After the database is created, restart this interface.
                """
            )
    
    return interface

# Launch the interface
if __name__ == "__main__":
    interface = create_gradio_interface()
    interface.launch(
        server_name="0.0.0.0",  # Allow external access
        server_port=7860,       # Default Gradio port
        share=False,            # Set to True if you want a public link
        debug=False,            # Set to True for debugging
        show_error=True         # Show errors in the interface
    )
