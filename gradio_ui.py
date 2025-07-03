import os
import gradio as gr
import re
from dotenv import load_dotenv
import time

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
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
# Initialize the RAG system
# -----------------------------
def initialize_rag_system():
    """Initialize the RAG system with existing Chroma database"""
    try:
        # Create Gemini embedding model
        embedding_model = GoogleGenerativeAIEmbeddings(
            model="models/embedding-001",
            google_api_key=gemini_api_key
        )
        
        # Load existing Chroma vectorstore
        persist_dir = "chroma_db"
        if not os.path.exists(persist_dir):
            raise FileNotFoundError(f"Chroma database not found at {persist_dir}. Please run the RAG setup script first.")
        
        vectorstore = Chroma(
            persist_directory=persist_dir,
            embedding_function=embedding_model
        )
        
        # Define Gemini chat model
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=gemini_api_key,
            temperature=0.1,
            max_output_tokens=1500
        )
        
        # Define improved RAG prompt with HTML formatting
        template = """You are an expert technical documentation assistant. Use the provided context to answer questions about the Genesys API documentation.

Instructions:
1. Format your response in clean, well-structured HTML with inline CSS styling
2. Use appropriate HTML tags: <h1>, <h2>, <h3> for headers, <ul>/<li> for lists, <pre><code> for code blocks
3. Apply inline CSS for professional styling (colors, fonts, spacing, borders)
4. Base your answer primarily on the provided context
5. If the context doesn't contain enough information, clearly state what information is missing
6. Provide specific details, examples, and step-by-step instructions when available
7. Use professional, technical language appropriate for developers
8. Make the HTML visually appealing with proper spacing, colors, and typography

Context:
{context}

Question: {question}

Please provide a comprehensive answer in HTML format with inline CSS styling:"""

        prompt = PromptTemplate(
            template=template,
            input_variables=["context", "question"]
        )
        
        # Create RAG chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            retriever=vectorstore.as_retriever(
                search_type="similarity",
                search_kwargs={"k": 5}
            ),
            chain_type="stuff",
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        return qa_chain, vectorstore
        
    except Exception as e:
        print(f"Error initializing RAG system: {e}")
        return None, None

# Initialize the system
qa_chain, vectorstore = initialize_rag_system()

# -----------------------------
# Query processing functions
# -----------------------------
def preprocess_query(query):
    """Preprocess the query to improve retrieval"""
    query = re.sub(r'\s+', ' ', query.strip())
    return query

def post_process_answer(answer):
    """Post-process the answer to improve HTML formatting"""
    # Clean up extra whitespace
    answer = re.sub(r'\n\s*\n\s*\n', '\n\n', answer)
    
    # Ensure proper HTML formatting
    answer = answer.strip()
    
    # Add some basic styling if the response doesn't have proper HTML structure
    if not answer.startswith('<') and not '<html>' in answer.lower():
        # Wrap plain text in a styled div
        answer = f'<div style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">{answer}</div>'
    
    return answer

# -----------------------------
# Main query function for Gradio
# -----------------------------
def query_documents(question, show_sources=False):
    """Main function to query the RAG system"""
    if not qa_chain:
        return '<div style="color: #dc3545; font-weight: bold;">❌ <strong>Error</strong>: RAG system not initialized. Please check your setup.</div>', ""
    
    if not question or not question.strip():
        return '<div style="color: #6c757d; font-style: italic;">❓ <strong>Please enter a question</strong> to get started.</div>', ""
    
    try:
        # Preprocess the query
        processed_query = preprocess_query(question)
        
        # Get the answer
        start_time = time.time()
        result = qa_chain.invoke(processed_query)
        end_time = time.time()
        
        # Post-process the answer
        answer = post_process_answer(result["result"])
        
        # Add processing time and metadata with HTML styling
        metadata = f'<div style="margin-top: 20px; padding: 10px; background: #f8f9fa; border-left: 4px solid #007bff; font-style: italic; color: #6c757d;">Response generated in {end_time - start_time:.2f} seconds</div>'
        
        # Prepare source information with HTML styling
        source_info = ""
        if show_sources and result.get("source_documents"):
            source_info = f'''
            <div style="margin-top: 20px; padding: 15px; background: #e9ecef; border-radius: 5px;">
                <h3 style="color: #495057; margin-top: 0;">📚 Source Information</h3>
                <p style="font-style: italic; color: #6c757d;">Answer based on {len(result['source_documents'])} relevant sections from the documentation</p>
            '''
            
            for i, doc in enumerate(result["source_documents"][:3], 1):
                # Show first 200 characters of each source
                content_preview = doc.page_content[:200].replace('\n', ' ').strip()
                source_info += f'''
                <div style="margin-bottom: 10px; padding: 10px; background: white; border-radius: 3px; border-left: 3px solid #28a745;">
                    <strong style="color: #28a745;">Source {i}:</strong> 
                    <span style="color: #495057;">{content_preview}...</span>
                </div>
                '''
            
            source_info += '</div>'
        
        full_response = f"{answer}{metadata}{source_info}"
        
        return full_response, ""
        
    except Exception as e:
        error_msg = f'<div style="color: #dc3545; padding: 15px; background: #f8d7da; border: 1px solid #f5c6cb; border-radius: 5px;"><strong>❌ Error processing your question:</strong> {str(e)}<br><br>Please try again with a different question.</div>'
        return error_msg, ""

# -----------------------------
# Sample questions
# -----------------------------
def load_sample_question(question):
    """Load a sample question"""
    return question

sample_questions = [
    "What is this document about?",
    "How do I authenticate with the API?",
    "What are the main API endpoints available?",
    "How do I handle errors in API responses?",
    "What are the rate limits for the API?",
    "How do I make a POST request to the API?",
    "What data formats are supported?",
    "How do I implement pagination?",
    "What are the security requirements?",
    "How do I troubleshoot API connection issues?"
]

# -----------------------------
# Gradio Interface
# -----------------------------
def create_gradio_interface():
    """Create the Gradio interface"""
    
    # Custom CSS for better styling
    custom_css = """
    .gradio-container {
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    .main-header {
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .sample-questions {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    """
    
    with gr.Blocks(css=custom_css, title="Genesys API Documentation Assistant") as demo:
        
        # Header
        gr.HTML("""
        <div class="main-header">
            <h1>APIGaido - Genesys API Documentation Assistant</h1>
            <p>Ask questions about the Genesys API documentation and get detailed, formatted answers</p>
        </div>
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                # Question input
                question_input = gr.Textbox(
                    label="Your Question",
                    placeholder="Enter your question about the Genesys API documentation...",
                    lines=3,
                    max_lines=5
                )
                
                # Options
                with gr.Row():
                    show_sources = gr.Checkbox(
                        label="📚 Show source information",
                        value=False
                    )
                    
                # Submit button
                submit_btn = gr.Button("🔍 Get Answer", variant="primary", size="lg")
                
                # Sample questions
                gr.HTML("""
                <div class="sample-questions">
                    <h3>💡 Sample Questions</h3>
                    <p>Click on any question below to try it out:</p>
                </div>
                """)
                
                # Create sample question buttons
                for i, question in enumerate(sample_questions):
                    sample_btn = gr.Button(f"📝 {question}", size="sm")
                    sample_btn.click(
                        fn=load_sample_question,
                        inputs=[gr.State(question)],
                        outputs=[question_input]
                    )
            
            with gr.Column(scale=3):
                # Answer output using HTML instead of Markdown
                answer_output = gr.HTML(
                    label="📋 Answer",
                    value='<div style="font-family: Arial, sans-serif; padding: 20px; background: #f8f9fa; border-radius: 5px; color: #495057;">Welcome! Enter a question to get started.</div>',
                )
                
                # Error output (hidden by default)
                error_output = gr.Textbox(
                    label="Error",
                    visible=False
                )
        
        # Usage instructions
        gr.HTML("""
        <div style="margin-top: 2rem; padding: 1rem; background: #e8f4f8; border-radius: 8px;">
            <h3>📖 How to Use</h3>
            <ul>
                <li><strong>Ask specific questions</strong> about API features, endpoints, authentication, etc.</li>
                <li><strong>Use clear, descriptive language</strong> for better results</li>
                <li><strong>Enable "Show source information"</strong> to see which parts of the documentation were used</li>
                <li><strong>Try the sample questions</strong> to get familiar with the system</li>
            </ul>
        </div>
        """)
        
        # Event handlers
        submit_btn.click(
            fn=query_documents,
            inputs=[question_input, show_sources],
            outputs=[answer_output, error_output]
        )
        
        question_input.submit(
            fn=query_documents,
            inputs=[question_input, show_sources],
            outputs=[answer_output, error_output]
        )
    
    return demo

# -----------------------------
# Launch the application
# -----------------------------
if __name__ == "__main__":
    # Check if the system is properly initialized
    if not qa_chain:
        print("❌ Failed to initialize RAG system. Please check your setup.")
        exit(1)
    
    print("✅ RAG system initialized successfully!")
    print("🚀 Launching Gradio interface...")
    
    # Create and launch the interface
    demo = create_gradio_interface()
    
    # Launch with custom settings
    demo.launch(
        share=False,  # Set to True if you want to create a public link
        server_name="0.0.0.0",  # Allow access from other devices on the network
        server_port=7860,  # Default Gradio port
        show_error=True
    )