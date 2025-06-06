# main.py

import os
from dotenv import load_dotenv
from utils import get_converter, save_converted_docs, load_docs, get_prompt_template

# Load environment variables
load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Constants
SOURCES = ["https://arxiv.org/pdf/2408.09869"]
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
QUESTION = "Which are the main AI models in Docling?"

def main():
    converter = get_converter()
    doc_store = save_converted_docs(converter, SOURCES)
    docs = load_docs(SOURCES, converter, EMBED_MODEL_ID)
    
    prompt = get_prompt_template()

    # Stub for future LLM/RAG integration
    print(f"Loaded {len(docs)} documents.")
    print(f"Prepared prompt:\n{prompt.format(context='...', input=QUESTION)}")

if __name__ == "__main__":
    main()
