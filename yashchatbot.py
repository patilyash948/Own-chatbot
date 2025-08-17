import os
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load .env
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Streamlit settings
st.set_page_config(page_title="Yash's Resume Chatbot", page_icon="🤖", layout="wide")
st.title("🤖 Yash's Resume Chatbot")
st.write("Ask anything based on **yash_resume.txt**")

# Cache vector store
@st.cache_resource
def load_vectorstore():
    loader = TextLoader("yash_resume.txt", encoding="utf-8")
    documents = loader.load()
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
    return FAISS.from_documents(documents, embeddings)

vectorstore = load_vectorstore()

# LLM
llm = OpenAI(temperature=0, openai_api_key=openai_api_key, max_tokens=150)

# Prompt
template = """
You are Yash's Resume Chatbot.
The context comes from an uploaded text file called 'yash_resume.txt', which contains Yash's resume.
Do not use 'my', 'I', 'me', or 'mine'. Provide the answer in Yash's name.
If asked about education, answer only based on the content of the uploaded file.
Always complete full sentences. If the token limit is reached, stop before exceeding it.
Always complete full sentences. If the token limit is reached, stop before exceeding it.
Always complete full sentences. If the token limit is reached, stop before exceeding it.(Always remember that)

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
    retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),
    chain_type="stuff",
    chain_type_kwargs={"prompt": prompt}
)

# Store chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# User input
user_input = st.chat_input("Type your question here...")

if user_input:
    # Add user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Get bot response
    with st.spinner("Thinking..."):
        bot_response = qa_chain.run(user_input)
    st.session_state.messages.append({"role": "assistant", "content": bot_response})

# Display messages in chat format
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])

if __name__ == "__main__":
    app.run()

