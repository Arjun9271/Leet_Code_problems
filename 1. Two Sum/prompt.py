from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import time
import os

# --- 1. Define your PDF file ---
# IMPORTANT: Replace 'sample.pdf' with the actual path to your PDF file.
# Make sure the PDF file exists in the same directory as your script, or provide the full path.
pdf_file_path = "sample.pdf"

# Create a dummy PDF for demonstration if you don't have one readily available.
# In a real scenario, you'd already have your PDF.
try:
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    def create_dummy_pdf(filename="sample.pdf"):
        c = canvas.Canvas(filename, pagesize=letter)
        c.drawString(100, 750, "This is the first page of the sample PDF.")
        c.drawString(100, 730, "It contains some basic information about AI and machine learning.")
        c.drawString(100, 710, "Artificial intelligence (AI) is a broad field focusing on intelligent machines.")
        c.showPage()
        c.drawString(100, 750, "The second page continues the discussion.")
        c.drawString(100, 730, "Machine learning, a subset of AI, involves training algorithms on data.")
        c.drawString(100, 710, "Deep learning uses neural networks for complex patterns.")
        c.save()
        print(f"Created a dummy PDF: {filename}")

    if not os.path.exists(pdf_file_path):
        create_dummy_pdf(pdf_file_path)
except ImportError:
    print("ReportLab not installed. Please manually create 'sample.pdf' or install ReportLab (`pip install reportlab`) to use the dummy PDF creator.")
    print("Make sure you have a 'sample.pdf' file ready for the script to use.")
    exit() # Exit if we can't create a dummy PDF and no PDF exists.


# --- 2. Load the PDF Document ---
print(f"Loading document from {pdf_file_path}...")
start_time = time.time()

loader = PyPDFLoader(pdf_file_path)
documents = loader.load()

# --- 3. Perform Recursive Chunking ---
print("Performing recursive chunking...")
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,  # Adjust chunk size based on your content and embedding model
    chunk_overlap=50, # Overlap helps maintain context between chunks
    length_function=len,
    is_separator_regex=False,
)
texts = text_splitter.split_documents(documents)

# --- 4. Create Embeddings ---
print("Creating embeddings...")
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- 5. Create FAISS Vector Database ---
print("Creating FAISS vector database...")
vectorstore = FAISS.from_documents(texts, embeddings)
print("FAISS vector database created.")

# --- 6. Perform a Retrieval Query (Direct Similarity Search) ---
print("\nPerforming a retrieval query (direct similarity search)...")
query = "What is machine learning?"
# We directly use the similarity_search method of the vectorstore
# k=4 means it will retrieve the top 4 most similar chunks
retrieved_docs = vectorstore.similarity_search(query, k=4)

print("\n--- Retrieval Results (Top K Chunks) ---")
print(f"Query: {query}")
print("\nRetrieved Documents:")
for i, doc in enumerate(retrieved_docs):
    print(f"--- Document {i+1} ---")
    print(doc.page_content)
    print(f"Source: {doc.metadata.get('source', 'N/A')}, Page: {doc.metadata.get('page', 'N/A')}")
    print("-" * 30)

# --- 7. Calculate and Print Time ---
end_time = time.time()
total_time = end_time - start_time
print(f"\nTotal execution time: {total_time:.2f} seconds")

# Clean up the dummy PDF file if it was created by the script
if 'create_dummy_pdf' in locals() and os.path.exists("sample.pdf"):
    os.remove("sample.pdf")
