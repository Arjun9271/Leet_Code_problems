# app.py
import os
import json
import streamlit as st
from pathlib import Path
from tempfile import mkdtemp
import matplotlib.pyplot as plt
from dotenv import load_dotenv
from PIL import ImageDraw
from PyPDF2 import PdfReader

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

# Init
st.set_page_config(layout="wide")
load_dotenv()
HF_TOKEN = os.getenv("HF_TOKEN")
EMBED_MODEL_ID = "sentence-transformers/all-MiniLM-L6-v2"
GEN_MODEL_ID = "meta-llama/Llama-3.1-8B-Instruct"
TOP_K = 3

st.title("📚 PDF Q&A with Visual Grounding")

# Sidebar: PDF Upload
st.sidebar.header("📤 Upload PDF")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    temp_pdf_path = Path("uploaded.pdf")
    with open(temp_pdf_path, "wb") as f:
        f.write(uploaded_file.read())

    # Convert PDF to Docling
    st.info("Processing PDF...")
    converter = DocumentConverter(format_options={
        InputFormat.PDF: PdfFormatOption(
            pipeline_options=PdfPipelineOptions(
                generate_page_images=True,
                images_scale=2.0,
            )
        )
    })
    dl_doc = converter.convert(str(temp_pdf_path)).document
    doc_store = {}
    file_path = Path(mkdtemp()) / f"{dl_doc.origin.binary_hash}.json"
    dl_doc.save_as_json(file_path)
    doc_store[dl_doc.origin.binary_hash] = file_path

    # Chunk and embed
    loader = DoclingLoader(
        file_path=[str(temp_pdf_path)],
        converter=converter,
        export_type="doc_chunks",
        chunker=HybridChunker(tokenizer=EMBED_MODEL_ID),
    )
    docs = loader.load()

    embedding = HuggingFaceEmbeddings(model_name=EMBED_MODEL_ID)
    vectorstore = Milvus.from_documents(
        documents=docs,
        embedding=embedding,
        collection_name="docling_demo",
        connection_args={"uri": str(Path(mkdtemp()) / "docling.db")},
        index_params={"index_type": "FLAT"},
        drop_old=True,
    )

    # QA + Visual Answer
    retriever = vectorstore.as_retriever(search_kwargs={"k": TOP_K})
    prompt = PromptTemplate.from_template(
        "Context information is below.\n---------------------\n{context}\n---------------------\nGiven the context information and not prior knowledge, answer the query.\nQuery: {input}\nAnswer:\n"
    )
    llm = HuggingFaceEndpoint(repo_id=GEN_MODEL_ID, huggingfacehub_api_token=HF_TOKEN)
    rag_chain = create_retrieval_chain(retriever, create_stuff_documents_chain(llm, prompt))

    # Q&A section
    col1, col2 = st.columns([1, 1.2])
    with col1:
        st.subheader("❓ Ask a Question")
        user_question = st.text_input("Enter your question")

        if st.button("Submit") and user_question:
            with st.spinner("💬 Generating answer..."):
                resp_dict = rag_chain.invoke({"input": user_question})
                st.markdown(f"**Answer:** {resp_dict['answer']}")

                # Create image highlights
                visual_dir = Path("visuals")
                visual_dir.mkdir(exist_ok=True)

                image_paths = []
                for i, doc in enumerate(resp_dict["context"]):
                    meta = DocMeta.model_validate(doc.metadata["dl_meta"])
                    dl_doc = DoclingDocument.load_from_json(doc_store[meta.origin.binary_hash])
                    image_by_page = {}

                    for doc_item in meta.doc_items:
                        if doc_item.prov:
                            prov = doc_item.prov[0]
                            page_no = prov.page_no
                            img = image_by_page.get(page_no)
                            if not img:
                                img = dl_doc.pages[page_no].image.pil_image
                                image_by_page[page_no] = img

                            bbox = prov.bbox.to_top_left_origin(dl_doc.pages[page_no].size.height).normalized(dl_doc.pages[page_no].size)
                            thickness = 2
                            padding = 2
                            bbox.l = round(bbox.l * img.width - padding)
                            bbox.r = round(bbox.r * img.width + padding)
                            bbox.t = round(bbox.t * img.height - padding)
                            bbox.b = round(bbox.b * img.height + padding)

                            draw = ImageDraw.Draw(img)
                            draw.rectangle(bbox.as_tuple(), outline="blue", width=thickness)

                    for page_no, img in image_by_page.items():
                        vis_path = visual_dir / f"vis_{i}_{page_no}.png"
                        img.save(vis_path)
                        image_paths.append(vis_path)

                st.session_state["images"] = image_paths
                st.session_state["img_index"] = 0

    # Right side: Visual grounding
    with col2:
        st.subheader("🖼️ Visual Answer")
        images = st.session_state.get("images", [])
        if images:
            index = st.session_state.get("img_index", 0)
            st.image(images[index], use_column_width=True)

            colA, colB = st.columns(2)
            if colA.button("⬅️ Previous"):
                if index > 0:
                    st.session_state["img_index"] = index - 1
            if colB.button("➡️ Next"):
                if index < len(images) - 1:
                    st.session_state["img_index"] = index + 1
        else:
            st.info("Answer image will appear here after asking a question.")
