import os
from dotenv import load_dotenv
from pypdf import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GEMINI_API_KEY")

# 1. Load and split PDF
def load_and_split_pdf(path: str) -> list[Document]:
    reader = PdfReader(path)
    texts = [page.extract_text() for page in reader.pages if page.extract_text()]
    raw = "\n".join(texts)

    # ✅ Use chunk_overlap, not overlap
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200
    )
    chunks = splitter.split_text(raw)

    return [Document(page_content=text) for text in chunks]

docs = load_and_split_pdf("en-GWE-8.1.2-API-book.pdf")

# 2. Create embeddings
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-exp-03-07"
)

# 3. Build vector store
vectordb = Chroma.from_documents(
    documents=docs,                 
    embedding=embeddings,
    persist_directory="./chroma_db",
    collection_name="rag_experiment"
)

retriever = vectordb.as_retriever(search_kwargs={"k": 3})

# 4. Build RAG chain
llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# 5. Query
query = "What is the AI Maturity Scale?"
result = qa_chain(query)
print("Answer:", result["result"])
for src in result["source_documents"]:
    print("Context chunk:", src.page_content[:200], "…")
