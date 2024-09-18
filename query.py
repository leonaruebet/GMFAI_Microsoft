import streamlit as st
from streamlit_chat import message
import os
from PIL import Image
from langchain.document_loaders import UnstructuredPDFLoader
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.text_splitter import TokenTextSplitter
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.callbacks import get_openai_callback
from dotenv import load_dotenv
import openai
import streamlit as st
import os
from PIL import Image
from PyPDF2 import PdfReader

load_dotenv()

# Function to get or set the API key
def get_api_key():
    api_key = os.getenv("OPENAI_API_KEY")  # Get the API key from environment variables
    if api_key is None:
        raise ValueError("API key is not set. Please check the configuration.")  # Raise an error if the API key is not found
    os.environ['OPENAI_API_KEY'] = api_key  # Set the API key in the environment
    openai.api_key = api_key  # Set the API key for OpenAI
    return api_key

def load_pdf(file_path):
    loader = UnstructuredPDFLoader(file_path)
    data = loader.load()
    return data

def process_pdf(file_path):
    data = load_pdf(file_path)
    text_splitter = TokenTextSplitter(chunk_size=50, chunk_overlap=0)
    doc = text_splitter.split_documents(data)
    
    embeddings = OpenAIEmbeddings()
    vectordb = Chroma.from_documents(doc, embedding=embeddings)
    
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    pdf_qa = ConversationalRetrievalChain.from_llm(OpenAI(), vectordb.as_retriever(), memory=memory)
    
    return pdf_qa

def main():
    st.set_page_config(layout="wide")
    st.header("[GMFAI] Mangrove Planning Assistant 🌴")

    api_key = get_api_key()
    if not api_key:
        st.warning("API key is not set. Please check the configuration.")
        return

    # Load the default PDF
    pdf_path = "./mangrove.pdf"
    if not os.path.exists(pdf_path):
        st.error(f"Default PDF file not found at {pdf_path}")
        return
    
    pdf_qa = process_pdf(pdf_path)
    # Chat interface
    st.subheader("Chat with the Mangrove Planning Assistant")
    
    if 'messages' not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])
        if "image" in message:
            st.image(message["image"])

    # File uploader for images
    with st.sidebar:
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
            with get_openai_callback() as cb:
                result = pdf_qa({"question": prompt})
                st.session_state.messages.append({"role": "assistant", "content": result['answer']})
                st.chat_message("assistant").markdown(result['answer'])
                st.info(f'Total Tokens: {cb.total_tokens}, Prompt Tokens: {cb.prompt_tokens}, Completion Tokens: {cb.completion_tokens}')

if __name__ == '__main__':
    main()
