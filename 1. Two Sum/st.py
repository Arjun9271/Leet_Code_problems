

import os
from pathlib import Path
import streamlit as st
from tempfile import mkdtemp
import json

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
from langchain_docling.loader import ExportType




os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Environment variables
HF_TOKEN = '"
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"


PROMPT = PromptTemplate.from_template(
    'context information is below.\ngiven the context information and not the prior knowledge ,answer the query.\nquery:{input}\nAnswer:\n'
)

TOP_K = 3
MILVUS_DIR = str(Path(mkdtemp()) / "docling.db")


uploaded_files = st.sidebar.file_uploader(
    'upload the pdf documents',
    type = ['pdf'],
    accept_multiple_files = True

)

SOURCES = []
if uploaded_files:
    tempdir = Path(mkdtemp())
    for file in uploaded_files:
        path = tempdir/file.name
        with open(path,'wb') as f:
            f.write(file.read())
        SOURCES.append(str(path))


# ──────────────────────────────────────────────────────────────────────────────
# 1) BUILD OR LOAD INDEX ONCE
# ──────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def build_vectorstore(sources):
    # 1a) converter: keep page images
    converter = DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=PdfPipelineOptions(
                    generate_page_images=True,
                    images_scale=2.0,
                )
            )
        }
    )
    # 1b) convert & save JSON per doc
    doc_json_store = {}
    tmp = Path(mkdtemp())
    for src in sources:
        res = converter.convert(src)
        dl_doc = res.document
        out = tmp / f"{dl_doc.origin.binary_hash}.json"
        dl_doc.save_as_json(out)
        doc_json_store[dl_doc.origin.binary_hash] = out

    # 1c) load chunks
    loader = DoclingLoader(
        file_path=sources,
        converter=converter,
        export_type=ExportType.DOC_CHUNKS,
        chunker=HybridChunker(tokenizer=EMBED_MODEL_ID),
    )
    docs = loader.load()

    # 1d) embed + store in Milvus
    embedder    = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)
    vectorstore = Milvus.from_documents(
        documents=docs,
        embedding=embedder,
        collection_name="streamlit_docling",
        connection_args={"uri": MILVUS_DIR},
        index_params={"index_type": "FLAT"},
        drop_old=True,
    )
    return vectorstore, doc_json_store

vectorstore, doc_json = build_vectorstore(SOURCES)
retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
llm       = HuggingFaceEndpoint(
    repo_id=GEN_MODEL_ID,
    huggingfacehub_api_token=HF_TOKEN
)
qa_chain = create_stuff_documents_chain(llm=llm, prompt=PROMPT)
rag_chain = create_retrieval_chain(
    llm=llm,
    retriever=retriever,
    combine_documents_chain=qa_chain,
)

# ──────────────────────────────────────────────────────────────────────────────
# 2) STREAMLIT LAYOUT
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(layout="wide")
cols = st.columns([3,2] if st.sidebar.checkbox("Chat on left", True) else [2,3])
chat_col, viz_col = cols

# Initialize session state
if "history" not in st.session_state:
    st.session_state.history = []    # list of (question, answer)
if "images" not in st.session_state:
    st.session_state.images = []     # list of PIL images per turn
if "img_idx" not in st.session_state:
    st.session_state.img_idx = 0

# Chat input
with chat_col:
    st.header("📄 Docling Chat")
    question = st.text_input("Ask a question about your docs:", "")
    if st.button("Submit", key="ask") and question:
        # Run RAG + grounding
        resp = rag_chain.invoke({"input": question})
        answer = resp["answer"]
        st.session_state.history.append((question, answer))

        # Build grounding images
        imgs = []
        for doc in resp["context"]:
            meta = DocMeta.model_validate(doc.metadata["dl_meta"])
            dl_doc = DoclingDocument.load_from_json(
                doc_json[meta.origin.binary_hash]
            )
            for item in meta.doc_items:
                if not item.prov: continue
                prov = item.prov[0]
                page = dl_doc.pages[prov.page_no]
                img = page.image.pil_image.copy()
                # normalized→pixel
                b = prov.bbox.to_top_left_origin(page_height=page.size.height)
                b = b.normalized(page.size)
                pad = 2
                x0 = round(b.l * img.width  - pad)
                y0 = round(b.t * img.height - pad)
                x1 = round(b.r * img.width  + pad)
                y1 = round(b.b * img.height + pad)
                draw = ImageDraw.Draw(img)
                draw.rectangle([x0,y0,x1,y1], outline="blue", width=2)
                imgs.append(img)
        st.session_state.images = imgs
        st.session_state.img_idx = 0

    # Display history
    for q,a in st.session_state.history:
        st.markdown(f"**You:** {q}")
        st.markdown(f"**Bot:** {a}")

# Visualization panel
with viz_col:
    st.header("📌 Grounding")
    imgs = st.session_state.images
    if not imgs:
        st.info("No grounding yet.")
    else:
        # Show current image
        st.image(imgs[st.session_state.img_idx], use_column_width=True)
        # Navigation
        if len(imgs) > 1:
            prev, nxt = st.columns(2)
            with prev:
                if st.button("◀️ Prev"):
                    st.session_state.img_idx = max(0, st.session_state.img_idx - 1)
            with nxt:
                if st.button("Next ▶️"):
                    st.session_state.img_idx = min(len(imgs)-1, st.session_state.img_idx + 1)


