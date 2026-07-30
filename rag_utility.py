import os
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_classic.chains import RetrievalQA




load_dotenv()

working_dir = os.path.dirname(os.path.abspath(__file__))

llm = ChatGroq(
    model = "openai/gpt-oss-120b",
    temperature = 0.1
)

embedding = HuggingFaceEmbeddings()

def process_document_to_chromadb(file_name):
    loader = PyPDFLoader(f"{working_dir}/{file_name}")
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024 , chunk_overlap= 200)
    text_chunks = text_splitter.split_documents(documents)

    vectordb = Chroma.from_documents(
        documents = text_chunks,
        embedding = embedding,
        persist_directory = f"{working_dir}/doc_vectorstore",
    )

    return 0

def answer_question(user_question):

 vectordb = Chroma(
        embedding_function= embedding,
        persist_directory=f"{working_dir}/doc_vectorstore",
    )

 retriever = vectordb.as_retriever()
 qa_chain = RetrievalQA.from_chain_type(
     llm=llm,
     retriever=retriever,
     chain_type="stuff"
   )

 response = qa_chain.invoke({"query": user_question})
 answer = response["result"]
 return answer

