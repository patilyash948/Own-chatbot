import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, OpenAI
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load environment variables
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Streamlit UI setup
st.set_page_config(page_title="Yash's Resume Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Resume Chatbot — Ask Questions Like You're Interviewing Yash")



# Cache vector store
@st.cache_resource
def load_vectorstore():
    
    loader = TextLoader("yash_resume.txt", encoding="utf-8")
    documents = loader.load()

    splitter = RecursiveCharacterTextSplitter(chunk_size=200, chunk_overlap=30)
    split_docs = splitter.split_documents(documents)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    return FAISS.from_documents(split_docs, embeddings)

vectorstore = load_vectorstore()

# LLM setup
llm = OpenAI(temperature=0, openai_api_key=openai_api_key, max_tokens=150)

# Prompt template
template = """
You are Yash's Resume Chatbot.
The context comes from an uploaded text file called 'yash_resume.txt', which contains Yash's resume.
Do not use 'my', 'I', 'me', or 'mine'. Provide the answer in Yash's name.
If asked about education, answer only based on the content of the uploaded file.
Always complete full sentences. If the token limit is reached, stop before exceeding it.

Context from yash_resume.txt:
{context}

Question: {question}
Answer:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 4}),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

# Chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# User input
user_input = st.chat_input("Type your question here...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.spinner("Thinking..."):
        bot_response = qa_chain.run(user_input)
    st.session_state.messages.append({"role": "assistant", "content": bot_response})

# Display chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
