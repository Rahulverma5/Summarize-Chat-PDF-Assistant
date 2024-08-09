import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS, Chroma
from langchain_community.embeddings import OllamaEmbeddings, GPT4AllEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.chat_models import ChatOllama
from htmlTemplates import css, bot_template, user_template

#function to extract text from all pdfs
def get_pdf_text(pdf_docs):
    text = ""                    #variable to store all text data in all pdfs
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

#function to get chuncks of data
def get_text_chunks(text)->list:
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=512,
        chunk_overlap=100,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks

#function to store embeddings in vectorstore
def get_vectorstore(text_chunks):
    embeddings = OllamaEmbeddings(model="all-minilm")
    vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
    return vectorstore

#this function takes history of conversation and return next conversation
def get_conversation_chain(vectorstore):
    llm = ChatOllama(model="llama2", temperature=0)
    memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain

def handle_userinput(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i%2 == 0:
            st.write(user_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True)
            



def main():
    load_dotenv() #used to store secret data in .env file
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    st.header("Chat with multiple PDFs :books:")

    #when user types a question
    user_question = st.text_input("Ask a question about your documents: ")
    if user_question:
        handle_userinput(user_question)

    # st.write(user_template.replace("{{MSG}}", "Hello Robot"), unsafe_allow_html=True)
    # st.write(bot_template.replace("{{MSG}}", "Hello Human"), unsafe_allow_html=True)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader("Upload your PDFs here :point_down:", accept_multiple_files=True)
        
        if st.button("Process"):
            with st.spinner("Processing"):
                #get pdf text
                raw_text = get_pdf_text(pdf_docs)
                #st.write(raw_text)

                #get the text chunks
                text_chunks = get_text_chunks(raw_text)
                #st.write(text_chunks)

                #create vectorstore
                vectorstore = get_vectorstore(text_chunks)
                
                #create conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)  #st.session_state helps to not reinitialize chats again after restart
                #this variable can be used outside of this button 


if __name__ == "__main__":
    main()