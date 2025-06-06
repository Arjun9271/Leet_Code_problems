# utils.py

import os
from pathlib import Path
from tempfile import mkdtemp
from langchain_core.prompts import PromptTemplate
from docling.datamodel.base_models import InputFormat
from docling.datamodel.pipeline_options import PdfPipelineOptions
from docling.document_converter import DocumentConverter, PdfFormatOption
from langchain_docling import DoclingLoader
from docling.chunking import HybridChunker
from langchain_docling.loader import ExportType

def get_converter():
    return DocumentConverter(
        format_options={
            InputFormat.PDF: PdfFormatOption(
                pipeline_options=PdfPipelineOptions(
                    generate_page_images=True,
                    images_scale=2.0,
                )
            )
        }
    )

def save_converted_docs(converter, sources):
    doc_store = {}
    doc_store_root = Path(mkdtemp())
    for source in sources:
        dl_doc = converter.convert(source=source).document
        file_path = doc_store_root / f"{dl_doc.origin.binary_hash}.json"
        dl_doc.save_as_json(file_path)
        doc_store[dl_doc.origin.binary_hash] = file_path
    return doc_store

def load_docs(sources, converter, embed_model_id):
    return DoclingLoader(
        file_path=sources,
        converter=converter,
        export_type=ExportType.DOC_CHUNKS,
        chunker=HybridChunker(tokenizer=embed_model_id),
    ).load()

def get_prompt_template():
    return PromptTemplate.from_template(
        """Context information is below.
---------------------
{context}
---------------------
Given the context information and not prior knowledge, answer the query.
Query: {input}
Answer:"""
    )
