import os
from pathlib import Path
from tempfile import mkdtemp
import json
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from PIL import ImageDraw

from langchain_core.prompts import PromptTemplate
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_docling import DoclingLoader
from langchain_huggingface import HuggingFaceEndpoint
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_milvus import Milvus

from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from docling.chunking import HybridChunker, DocMeta
from docling.datamodel.document import DoclingDocument

load_dotenv()
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Environment variables
HF_TOKEN = os.getenv("HF_TOKEN")

# Constants
SOURCES = ["example.pdf"]
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
QUESTION = "Which is famous 7B model"
TOP_K = 3

# Prompt
PROMPT = PromptTemplate.from_template(
    "Context information is below.\n---------------------\n{context}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {input}\nAnswer:\n",
)

# Convert PDF to Docling format
converter = DocumentConverter(
    format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=PdfPipelineOptions(
                generate_page_images=True,
                images_scale=2.0,
            ),
        )
    }
)

# Store converted docs
doc_store = {}
doc_store_root = Path(mkdtemp())
for source in SOURCES:
    dl_doc = converter.convert(source=source).document
    file_path = Path(doc_store_root / f"{dl_doc.origin.binary_hash}.json")
    dl_doc.save_as_json(file_path)
    doc_store[dl_doc.origin.binary_hash] = file_path

# Load and chunk docs
loader = DoclingLoader(
    file_path=SOURCES,
    converter=converter,
    export_type="doc_chunks",
    chunker=HybridChunker(tokenizer=EMBED_MODEL_ID),
)
docs = loader.load()

# Print token counts
for d in docs:
    print('Token len:', len(d.page_content.split(" ")))
    print(f'- {d.page_content}')
print('.....')

# Embed and store vectors in Milvus
embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)
milvus_uri = str(Path(mkdtemp()) / "docling.db")
vectorstore = Milvus.from_documents(
    documents=docs,
    embedding=embedding,
    collection_name="docling_demo",
    connection_args={"uri": milvus_uri},
    index_params={"index_type": "FLAT"},
    drop_old=True,
)

# Create LLM and RAG chain
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
llm = HuggingFaceEndpoint(
    repo_id=GEN_MODEL_ID,
    huggingfacehub_api_token=HF_TOKEN,
)
question_answer_chain = create_stuff_documents_chain(llm, PROMPT)
rag_chain = create_retrieval_chain(retriever, question_answer_chain)
resp_dict = rag_chain.invoke({"input": QUESTION})

# Print result
def clip_text(text, threshold=100):
    return f"{text[:threshold]}..." if len(text) > threshold else text

clipped_answer = clip_text(resp_dict["answer"], threshold=250)
print(f"Question:\n{resp_dict['input']}\n\nAnswer:\n{clipped_answer}")

# Visualize answers
for i, doc in enumerate(resp_dict["context"]):
    image_by_page = {}
    print(f"Source {i + 1}:")
    print(f"  text: {json.dumps(clip_text(doc.page_content, threshold=350))}")
    meta = DocMeta.model_validate(doc.metadata["dl_meta"])

    dl_doc = DoclingDocument.load_from_json(doc_store.get(meta.origin.binary_hash))

    for doc_item in meta.doc_items:
        if doc_item.prov:
            prov = doc_item.prov[0]
            page_no = prov.page_no
            if img := image_by_page.get(page_no):
                pass
            else:
                page = dl_doc.pages[prov.page_no]
                print(f"  page: {prov.page_no}")
                img = page.image.pil_image
                image_by_page[page_no] = img
            bbox = prov.bbox.to_top_left_origin(page_height=page.size.height)
            bbox = bbox.normalized(page.size)
            thickness = 2
            padding = thickness + 2
            bbox.l = round(bbox.l * img.width - padding)
            bbox.r = round(bbox.r * img.width + padding)
            bbox.t = round(bbox.t * img.height - padding)
            bbox.b = round(bbox.b * img.height + padding)
            draw = ImageDraw.Draw(img)
            draw.rectangle(xy=bbox.as_tuple(), outline="blue", width=thickness)
    for p in image_by_page:
        img = image_by_page[p]
        plt.figure(figsize=[15, 15])
        plt.imshow(img)
        plt.axis("off")
        plt.show()
