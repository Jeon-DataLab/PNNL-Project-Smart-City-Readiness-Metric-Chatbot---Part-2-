# -*- coding: utf-8 -*-


import os
import fitz
from haystack import Document
from haystack.document_stores import InMemoryDocumentStore
from haystack.document_stores import ElasticsearchDocumentStore
from haystack.utils import convert_files_to_docs
from haystack.nodes import BM25Retriever
from flask import Flask, request, render_template
from flask_ngrok import run_with_ngrok
import streamlit as st
from pyngrok import ngrok
document_store = InMemoryDocumentStore(use_bm25=True)

from haystack.nodes import FARMReader

from haystack.pipelines import ExtractiveQAPipeline

from haystack.utils import print_answers

# Path to the directory containing the PDF files
pdf_directory = "/Users/sohinigudapati/Documents/Mourya/Work/Resilience/"
from sentence_transformers import CrossEncoder, SentenceTransformer
# Function to extract text from PDFs
def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Iterate over PDF files in the directory
def read_all_files():
  for pdf_file in os.listdir(pdf_directory):
      if pdf_file.endswith(".pdf"):
          pdf_path = os.path.join(pdf_directory, pdf_file)
          text = extract_text_from_pdf(pdf_path)
          # Create a Document object for each PDF using the Document constructor provided by Haystack
          doc = Document(content=text, meta={"name": pdf_file})
          # Write the Document to the InMemoryDocumentStore
          document_store.write_documents([doc])

pdf_text = read_all_files()
retriever = BM25Retriever(document_store=document_store)
reader = FARMReader(model_name_or_path="deepset/roberta-base-squad2")
pipe = ExtractiveQAPipeline(reader, retriever)

def main():
    st.title("City Resilience QA App")

    # Initialize a sample pipe object (replace with your actual initialization)
    question = st.text_input("Ask a question about city resilience:")

    if st.button("Get Answer"):
        prediction = pipe.run(query=question, params={"Retriever": {"top_k": 10}, "Reader": {"top_k": 5}})
        if prediction and prediction['answers']:
            i = 0
            for answer in prediction["answers"]:
                
                answer_text = answer.answer
                # answer_text = answer['answer'][0]['answer']
                st.write("Answer:", i, str(answer_text), "\n")
                i+=1
            print_answers(prediction, details="minimum")
        else:
            st.write("Sorry, couldn't find an answer.")

if __name__ == '__main__':
    main()



