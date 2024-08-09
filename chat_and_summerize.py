import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama
from transformers import pipeline  # Import pipeline for summarization
from htmlTemplates import css, bot_template, user_template

# Function to extract text from all PDFs
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

# Function to get chunks of data
def get_text_chunks(text) -> list:
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=512,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

# Function to store embeddings in vectorstore
def get_vectorstore(text_chunks):
    embeddings = OllamaEmbeddings(model="all-minilm")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

# Function to get conversation chain
def get_conversation_chain(vectorstore):
    llm = ChatOllama(model="llama2", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

# Function to summarize text
def summarize_text(text):
    summarizer = pipeline("summarization")
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Function to handle user input
def handle_userinput(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        template = user_template if i % 2 == 0 else bot_template
        st.write(template.replace("{{MSG}}", message.content), unsafe_allow_html=True)

def main():
    load_dotenv()  # Load environment variables
    st.set_page_config(page_title="Study Assistant", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Study Assistant :books:")

    # User question input
    user_question = st.text_input("Ask a question about your study materials:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your Study Materials")
        pdf_docs = st.file_uploader("Upload your PDFs here :point_down:", accept_multiple_files=True, type=["pdf"])
        
        if st.button("Process"):
            with st.spinner("Processing..."):
                try:
                    # Get PDF text
                    raw_text = get_pdf_text(pdf_docs)

                    # Get text chunks
                    text_chunks = get_text_chunks(raw_text)

                    # Create vectorstore
                    vectorstore = get_vectorstore(text_chunks)
                
                    # Create conversation chain
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.success("Documents processed successfully!")
                except Exception as e:
                    st.error(f"An error occurred: {e}")

        if st.button("Summarize Documents"):
            with st.spinner("Summarizing..."):
                try:
                    # Get PDF text
                    raw_text = get_pdf_text(pdf_docs)
                    
                    # Summarize text
                    summary = summarize_text(raw_text)
                    st.write("### Summary of Documents")
                    st.write(summary)
                except Exception as e:
                    st.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
