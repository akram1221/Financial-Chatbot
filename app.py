import streamlit as st
import pandas as pd
import fitz  # PyMuPDF
import os
import json
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain.vectorstores import FAISS
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from concurrent.futures import ThreadPoolExecutor

# Set page config
st.set_page_config(page_title="Fintellia")

# Load API key from Streamlit Secrets
API_KEY = st.secrets["GOOGLE_API_KEY"]["value"]

# Debug: Print API key to ensure it is being retrieved correctly
st.write(f"API_KEY: {API_KEY}")

# Configure genai with the API key
genai.configure(api_key=API_KEY)

PDF_DIRECTORY = "FinancialDocs1"  # Update this with the actual directory path
CSV_FILE_PATH = "monthly_stock_prices_2019_2023_yf.csv"  # Update this with the actual CSV file path
CACHE_FILE = "pdf_text_cache.json"
INDEX_FILE = "faiss_index"

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
    with ThreadPoolExecutor() as executor:
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
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    chunks = text_splitter.split_text(text)
    return chunks

def get_vector_store(text_chunks):
    try:
        embeddings = GoogleGenerativeAIEmbeddings(google_api_key=API_KEY, model="models/embedding-001")
        vector_store = FAISS.from_texts(text_chunks, embeddings)
        vector_store.save_local(INDEX_FILE)
        return vector_store
    except Exception as e:
        st.error(f"Error in embedding text chunks: {e}")
        raise

def load_vector_store():
    try:
        embeddings = GoogleGenerativeAIEmbeddings(google_api_key=API_KEY, model="models/embedding-001")
        vector_store = FAISS.load_local(INDEX_FILE, embeddings, allow_dangerous_deserialization=True)
        return vector_store
    except Exception as e:
        st.error(f"Error loading vector store: {e}")
        raise

def get_conversational_chain():
    prompt_template = """
    You are a highly intelligent and detail-oriented financial assistant. Follow the steps below to answer the user's financial question about a specific company:

    User Question: {question}

    You are an intelligent financial chatbot designed to assist users with queries related to financial reports, stock prices, investment decisions, and company performance. Your primary goal is to provide accurate, timely, and insightful answers based on the latest financial data available from reliable sources like SEC EDGAR filings and stock market data.

    When responding to queries, follow these guidelines:

    1. **Understand the Question**: Carefully analyze the user's question to determine what specific financial information they are seeking. Clarify any ambiguous queries by asking follow-up questions if necessary.
    2. **Use Reliable Sources**: Base your responses on the most recent and relevant financial data available. Reference SEC EDGAR filings, particularly form 10-K for comprehensive annual reports, and check the CSV file for current stock prices from reliable financial databases.
    3. **Provide Clear and Concise Answers**: Aim to deliver answers that are easy to understand, free from jargon, and directly address the user's query. Include relevant data points, comparisons, and explanations as needed.
    4. **Offer Additional Insights**: Where appropriate, provide additional context or insights that may help the user make informed decisions. This can include trends, historical data, and potential future implications.
    5. **Stay Neutral and Objective**: Maintain an impartial tone, avoiding any bias or subjective opinions. Present the facts and data as they are, without suggesting personal recommendations or advice.
    6. **Ensure Accuracy and Consistency**: Double-check the accuracy of the data and information you provide. Ensure consistency in your answers, especially when similar queries are asked.

    **Disclaimer**: The information provided is based on the latest available financial filings and stock data. This may not reflect the most current data. Consult the most recent filings for up-to-date information.

    Context:
    {context}

    Answer:
    """
    model = ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.3)
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    chain = load_qa_chain(model, chain_type="stuff", prompt=prompt)
    return chain

def user_input(user_question):
    try:
        vector_store = load_vector_store()
        docs = vector_store.similarity_search(user_question)
        chain = get_conversational_chain()
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        return response["output_text"]
    except Exception as e:
        st.error(f"Error processing user input: {e}")
        raise

def process_files():
    try:
        if not os.path.exists(INDEX_FILE):
            with ThreadPoolExecutor() as executor:
                pdf_future = executor.submit(get_pdf_text_from_directory, PDF_DIRECTORY)
                csv_future = executor.submit(get_csv_text, CSV_FILE_PATH)

                pdf_text = pdf_future.result()
                csv_text = csv_future.result()

            combined_text = pdf_text + csv_text
            text_chunks = get_text_chunks(combined_text)
            
            # Embed text chunks with rate limiting
            get_vector_store(text_chunks)
        else:
            load_vector_store()
    except Exception as e:
        st.error(f"Error processing files: {e}")
        raise

def main():
    st.title("Financial Q&A Chatbot")

    st.markdown("""
        <style>
        .user-message {
            text-align: right;
            background-color: #e1f5fe;
            border-radius: 10px;
            padding: 10px;
            margin: 10px 0;
            display: flex;
            justify-content: flex-end;
            max-width: 80%;
            margin-left: auto;
        }
        .assistant-message {
            text-align: left;
            background-color: #ffffff;
            border-radius: 10px;
            padding: 10px;
            margin: 10px 0;
            display: flex;
            justify-content: flex-start;
            max-width: 80%;
            margin-right: auto;
        }
        .stTextInput > div > div {
            background-color: transparent;
        }
        .stTextInput textarea {
            border-radius: 10px;
        }
        .chat-icon {
            width: 24px;
            height: 24px;
            margin: 0 10px;
        }
        .loader {
          width: 30px;
          aspect-ratio: 2;
          --_g: no-repeat radial-gradient(circle closest-side,#000 90%,#0000);
          background: 
            var(--_g) 0%   50%,
            var(--_g) 50%  50%,
            var(--_g) 100% 50%;
          background-size: calc(100%/3) 50%;
          animation: l3 1s infinite linear;
        }
        @keyframes l3 {
            20%{background-position:0%   0%, 50%  50%,100%  50%}
            40%{background-position:0% 100%, 50%   0%,100%  50%}
            60%{background-position:0%  50%, 50% 100%,100%   0%}
            80%{background-position:0%  50%, 50%  50%,100% 100%}
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
