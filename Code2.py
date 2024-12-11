import os
import time
from dotenv import load_dotenv
import streamlit as st

# LangChain community modules for loading PDFs and using embeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# LangChain core components for handling prompts and chains
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain


# Load environment variables from .env file (API keys, etc.)
load_dotenv()
os.environ['HUGGINGFACE_API_KEY'] = os.getenv("HUGGINGFACE_API_KEY")
groq_api_key = os.getenv("GROQ_API_KEY")

# Initializing the LLM (Large Language Model) with Groq's API using Llama3 model
llm = ChatGroq(model="Llama3-8b-8192", api_key=groq_api_key)

# Defining the prompt template to guide the LLM in generating responses
prompt = ChatPromptTemplate.from_template(
    """
    Generate the most accurate responses to the questions using only the provided context. 
    Ensure that your answers are strictly based on the information given.
    <context>
    {context}
    <context>
    Question: {input}
    """
)

# Function to create vector stores (data ingestion, transformation, embedding)
def create_vector_stores():
    if "vectors" not in st.session_state:
        print("Step 1 ---------->>>>> Data Ingestion Starting")
        # Loading PDF documents from the specified directory
        st.session_state.loader = PyPDFDirectoryLoader("10_RAG_Q&A_LLAMA3_APP/PDFs")
        st.session_state.docs = st.session_state.loader.load() 
        print("Step 1 ---------->>>>> Data Ingestion Done\n")

        print("Step 2 ---------->>>>> Data Transformation Starting")
        # Splitting the documents into smaller chunks for better processing
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(st.session_state.docs)
        print("Step 2 ---------->>>>> Data Transformation Done\n")

        print("Step 3 ---------->>>>> Vector Embedding & Vector Store Starting")
        # Converting text documents into vector embeddings using a pre-trained Hugging Face model
        st.session_state.embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
        # Storing these embeddings in a FAISS vector store for efficient retrieval
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents, st.session_state.embeddings)
        print("Step 3 ---------->>>>> Vector Embedding & Vector Store Done\n")
         
# Streamlit UI title
st.title("RAG Document Q&A With Groq And Llama3")
# Text input for the user's query
user_prompt = st.text_input("Enter your query from the provided pdfs")

# Button to initiate the document embedding process
if st.button("Document Embedding"):
    create_vector_stores()
    st.write("Vector Database Is Ready. Now You Can Ask Any Question From The RAG Model From The Given PDFs.")

# If a user query is entered, we proceed with the retrieval and response generation
if user_prompt:
    # Creating a document chain that uses our prompt and the LLM
    document_chain = create_stuff_documents_chain(llm, prompt)
    # Setting up a retriever from our vector store to find relevant documents
    retriever = st.session_state.vectors.as_retriever()
    # Creating the retrieval chain that links our retriever and document chain
    retrieval_chain = create_retrieval_chain(retriever, document_chain)

    # Measuring response time for the query
    start = time.process_time()
    # Running the retrieval chain to get an answer
    response = retrieval_chain.invoke({'input': user_prompt})
    print(f"Response time : {time.process_time()-start}")

    # Displaying the final answer in the Streamlit app
    st.write(response['answer'])

    # Optionally showing the documents that were found to be similar to the query
    with st.expander("Document similarity Search"):
        for i, doc in enumerate(response['context']):
            st.write(doc.page_content)
            st.write('------------------------')