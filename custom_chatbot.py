import streamlit as st
from streamlit_chat import message
import os
from dotenv import load_dotenv
import openai  # Ensure this line is present
from PyPDF2 import PdfReader
from PIL import Image
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.llms import OpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.prompts import PromptTemplate
import streamlit as st
import os
from PIL import Image
from PyPDF2 import PdfReader

# At the top of your Python files
from dotenv import load_dotenv
import os

load_dotenv()  # Load environment variables from .env file
api_key = os.getenv("OPENAI_API_KEY")

def get_api_key():
    api_key = os.getenv("OPENAI_API_KEY")  # Get the API key from environment variables
    if api_key is None:
        raise ValueError("API key is not set. Please check the configuration.")  # Raise an error if the API key is not found
    os.environ['OPENAI_API_KEY'] = api_key  # Set the API key in the environment
    openai.api_key = api_key  # Set the API key for OpenAI
    return api_key

def extract_text_from_pdf(pdf_path):
    with open(pdf_path, 'rb') as file:
        pdf_reader = PdfReader(file)
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def process_text(text):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000,
        chunk_overlap = 100,
        length_function = len
    )
    chunks = text_splitter.split_text(text=text)
    return chunks

def main():
    st.header("Mangrove Planning Assistant 🌴")

    api_key = get_api_key()
    if not api_key:
        st.warning("API key is not set. Please check the configuration.")
        return

    # Load the default PDF
    pdf_path = "./mangrove.pdf"
    if not os.path.exists(pdf_path):
        st.error(f"Default PDF file not found at {pdf_path}")
        return
    
    text = extract_text_from_pdf(pdf_path)
    chunks = process_text(text)

    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    VectorStore = FAISS.from_texts(texts=chunks, embedding=embeddings)

    # Set up conversation memory
    if 'buffer_memory' not in st.session_state:
        st.session_state.buffer_memory = ConversationBufferWindowMemory(memory_key='chat_history', k=3, return_messages=True)

    # Create a custom prompt template
    template = """You are an AI assistant specialized in mangrove plants who can help with planning and benchmarking. 
    Use the following files to answer the human's question about mangroves, mangrove planning and benchmarking. 
    If the human has uploaded an image, they may ask questions about it in relation to mangrove planning and benchmarking.  
    If you don't know the answer, make a confident answer up.
    If user ask about how good the location is, always make a benchmake score from 0 - 100
    If user ask in thai languge answer in thai language
    {context}
    
    Human: {question}
    AI Assistant:"""

    PROMPT = PromptTemplate(template=template, input_variables=["context", "question"])

    # Set up the conversation chain
    conversation = ConversationalRetrievalChain.from_llm(
        llm=OpenAI(temperature=0.7),
        retriever=VectorStore.as_retriever(),
        memory=st.session_state.buffer_memory,
        combine_docs_chain_kwargs={"prompt": PROMPT}
    )
    
    # Chat interface
    st.subheader("Chat with the Mangrove Planning Assistant")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])
        if "image" in message:
            st.image(message["image"])

    # File uploader for images
    uploaded_file = st.file_uploader("Upload an image related to mangrove planning (optional)", type=["png", "jpg", "jpeg"])
    
    if prompt := st.chat_input("What would you like to know about mangrove benchmarks and planning?"):
        st.session_state.messages.append({"role": "user", "content": prompt})
        st.chat_message("user").markdown(prompt)

        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            st.session_state.messages[-1]["image"] = image
            prompt += "\n\nNote: An image has been uploaded. The user may ask questions about it."

        with st.spinner("Thinking..."):
            response = conversation({"question": prompt})
            st.session_state.messages.append({"role": "assistant", "content": response['answer']})
            st.chat_message("assistant").markdown(response['answer'])

if __name__ == '__main__':
    main()