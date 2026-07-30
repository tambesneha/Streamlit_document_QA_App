import os
import streamlit as st
from rag_utility import process_document_to_chromadb , answer_question

working_dir = os.path.dirname(os.path.abspath(__file__))

st.title("LLAMA-OpenAI-Document RAG")

uploaded_file = st.file_uploader("Upload a PDF file" , type=['pdf'])

if uploaded_file is not None:
    save_path = os.path.join(working_dir , uploaded_file.name)

    with open(save_path, 'wb') as f:
           f.write(uploaded_file.getbuffer())

    process_document = process_document_to_chromadb(uploaded_file.name)
    st.info("Document processed successfully")

user_question = st.text_area("Ask your question about document")

if st.button("Answer"):
    answer = answer_question(user_question)

    st.markdown("### Openai-120 response")
    st.markdown(answer)

