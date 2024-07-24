import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import os
import json
import time
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import concurrent.futures

# Hardcode your API key here
API_KEY = "AIzaSyDHOEnTdvKYz5OzuLGm4KafHS6fAcLeVDg"

genai.configure(api_key=API_KEY)

PDF_DIRECTORY = "Financial docs"  # Update this with the actual directory path
CSV_FILE_PATH = "monthly_stock_prices_2019_2023_yf.csv"  # Update this with the actual relative path to your CSV file
CACHE_FILE = "pdf_text_cache.json"

def extract_text_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text

def load_cached_text():
    if os.path.exists(CACHE_FILE):
        with open(CACHE_FILE, "r") as f:
            return json.load(f)
    return None

def save_cached_text(text):
    with open(CACHE_FILE, "w") as f:
        json.dump(text, f)

def get_pdf_text_from_directory(directory):
    cached_text = load_cached_text()
    if cached_text:
        return cached_text

    pdf_paths = [os.path.join(directory, filename) for filename in os.listdir(directory) if filename.endswith(".pdf")]

    texts = []
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(extract_text_from_pdf, pdf_path) for pdf_path in pdf_paths]
        for future in concurrent.futures.as_completed(futures):
            texts.append(future.result())

    combined_text = "".join(texts)
    save_cached_text(combined_text)
    return combined_text

def get_csv_text(file_path):
    df = pd.read_csv(file_path)
    text = df.to_string(index=False)
    return text

def get_text_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    chunks = text_splitter.split_text(text)
    return chunks

def embed_text_chunks(chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(api_key=API_KEY)
        vectorstore = FAISS.from_texts(chunks, embeddings)
        return vectorstore
    except Exception as e:
        st.error(f"Error in embedding text chunks: {e}")
        raise

def process_files():
    pdf_text = get_pdf_text_from_directory(PDF_DIRECTORY)
    csv_text = get_csv_text(CSV_FILE_PATH)
    combined_text = pdf_text + "\n" + csv_text
    chunks = get_text_chunks(combined_text)
    vectorstore = embed_text_chunks(chunks)
    st.session_state.vectorstore = vectorstore

def user_input(prompt):
    vectorstore = st.session_state.vectorstore
    qa_chain = load_qa_chain(vectorstore)
    result = qa_chain.run(prompt)
    return result

def main():
    st.set_page_config(page_title="Financial Q&A Chatbot", layout="wide")
    st.title("Financial Q&A Chatbot")

    st.markdown("""
        <style>
        .user-message {
            background-color: #dcf8c6;
            border-radius: 10px;
            padding: 10px;
            margin: 5px 0;
            max-width: 60%;
            align-self: flex-end;
            word-wrap: break-word;
        }
        .assistant-message {
            background-color: #ececec;
            border-radius: 10px;
            padding: 10px;
            margin: 5px 0;
            max-width: 60%;
            align-self: flex-start;
            word-wrap: break-word;
        }
        .chat-icon {
            margin-right: 10px;
        }
        .loader {
            border: 4px solid #f3f3f3;
            border-top: 4px solid #3498db;
            border-radius: 50%;
            width: 20px;
            height: 20px;
            animation: spin 2s linear infinite;
        }
        @keyframes spin {
            0% { transform: rotate(0deg); }
            100% { transform: rotate(360deg); }
        }
        .chat-container {
            display: flex;
            flex-direction: column;
            max-height: 70vh;
            overflow-y: auto;
            padding: 10px;
        }
        </style>
    """, unsafe_allow_html=True)

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.get("files_processed"):
        process_files()
        st.session_state["files_processed"] = True

    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    for message in st.session_state.messages:
        align_class = "user-message" if message["role"] == "user" else "assistant-message"
        icon = "👤" if message["role"] == "user" else "💬"
        st.markdown(f'<div class="{align_class}"><span class="chat-icon">{icon}</span>{message["content"]}</div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    if prompt := st.chat_input("Ask your financial question here..."):
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        placeholder = st.empty()
        with placeholder.container():
            st.markdown('<div class="assistant-message"><span class="chat-icon">💬</span><div class="loader"></div></div>', unsafe_allow_html=True)

        response = user_input(prompt)
        placeholder.empty()
        with st.chat_message("assistant"):
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == "__main__":
    main()
