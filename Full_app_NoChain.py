import streamlit as st
import os
import pickle
from dotenv import load_dotenv
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from htmlTemplates import css, bot_template, user_template
from langchain.llms import HuggingFaceHub

cache_dir="db"



def get_python_code_info(py_files):
    code_info = []
    for file_path in py_files:
        with open(file_path, "r", encoding="utf-8") as file:
            file_content = file.read()
            code_info.append((file_path, file_content))
    return code_info


def get_code_chunks(code_info):
    chunks = []
    for info in code_info:
        chunks.append(info)
    return chunks

#def get_code_chunks(text):
#  chunks = [] 
#  for chunk in text:
#    text_splitter = CharacterTextSplitter(
#        separator="\n",
#        chunk_size=100,
#        chunk_overlap=20,
#        length_function=len
#    )
#    chunks += text_splitter.split_text(chunk)
#    return chunks

def get_cached_data():
    cache_file = os.path.join(cache_dir, "cached_data.pkl")
    try:
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    except FileNotFoundError:
        return None


def save_cached_data(vectorstore, chunks):
    cache_file = os.path.join(cache_dir, "cached_data.pkl")
    with open(cache_file, "wb") as f:
        pickle.dump({"vectorstore": vectorstore, "chunks": chunks}, f)


def get_vectorstore(text_chunks):
    embeddings = OpenAIEmbeddings()
    texts = []
    for chunk in text_chunks:
        file_path, code_content = chunk
        text = file_path + "\n" + code_content
        texts.append(text)
    vectorstore = FAISS.from_texts(texts=texts, embedding=embeddings)
    return vectorstore



def get_conversation_chain(vectorstore):
    llm = ChatOpenAI()
    # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain


def handle_userinput(user_question):
    response = st.session_state.conversation({'question': user_question})
    st.session_state.chat_history = response['chat_history']

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(user_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)
        else:
            st.write(bot_template.replace(
                "{{MSG}}", message.content), unsafe_allow_html=True)


def main():
    load_dotenv()
    st.write(css, unsafe_allow_html=True)
    cached_data = get_cached_data()
    # Check for cached data and use it if available
    if "conversation" not in st.session_state:
        st.session_state.conversation = None

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.subheader("Your PY Files")


    st.header("Chat with multiple PY files :books:")
    user_question = st.text_input("Ask a question about your documents:")
    db_files = os.listdir(cache_dir)

    if user_question:
      handle_userinput(user_question)
    if db_files:
        cached_data = get_cached_data()
        vectorstore, chunks = cached_data["vectorstore"], cached_data["chunks"]
        st.session_state.conversation = get_conversation_chain(vectorstore)
        #history = []
        #handle_userinput(user_question)
        #history = history.append(st.session_state.chat_history)
        #st.write(history)

    else:
     with st.sidebar:
      #try:
        path_input = st.text_input("Enter the path to the directory containing Python files:")
        #handle_previous_inputs = st.checkbox("Handle Previous Inputs")
        if path_input:
            if os.path.exists(path_input) and os.path.isdir(path_input):
                py_files = []
                for root, dirs, files in os.walk(path_input):
                    for file in files:
                        if file.endswith('.py') or file.endswith('.js') or file.endswith('.xml'):
                            file_path = os.path.join(root, file)
                            py_files.append(file_path)
                            st.write(f"Added file to py_files: {file_path}")
            else:
                raise FileNotFoundError("Invalid directory path. Please enter a valid path.")
      #except Exception as e:
       # st.error(f"Error: {str(e)}")

        #if cached_data:
        #    vectorstore, code_chunks = cached_data["vectorstore"], cached_data["code_chunks"]
        #    st.session_state.conversation = get_conversation_chain(vectorstore)
        #    history = []
        #    handle_userinput(user_question)
        #    history.append(st.session_state.chat_history)
        if st.button("Handel New Files"):
         with st.spinner("Processing Python code..."):
             raw_code_info = get_python_code_info(py_files)
          #st.write("file Content")
          #st.write(raw_code_info)
             code_chunks = get_code_chunks(raw_code_info)
             text_chunks_with_paths = [(chunk[0], chunk[1]) for chunk in code_chunks]
          #st.write("chunk is")
          #st.write(code_chunks)
             vectorstore = get_vectorstore(text_chunks_with_paths)
          #st.write("vector is ")
          #st.write(vectorstore)
          #conversation = get_conversation_chain(vectorstore)
          #oconversation = get_conversation_chain(vectorstore)
             save_cached_data(vectorstore, code_chunks)
             st.session_state.conversation = get_conversation_chain(vectorstore)
          #save_cached_data(vectorstore, code_chunks)  # Cache processed data
          #if handle_previous_inputs and cached_data:
          # pass

if __name__ == '__main__':
    main()
