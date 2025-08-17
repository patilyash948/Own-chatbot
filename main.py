import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# Load .env
load_dotenv()

# Get OpenAI API key
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load resume
loader = TextLoader("yash_resume.txt", encoding="utf-8")
documents = loader.load()

# Create vector store
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vectorstore = FAISS.from_documents(documents, embeddings)

# Load LLM
llm = OpenAI(temperature=0, openai_api_key=openai_api_key, max_tokens=80)

# Create QA chain
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

# Chat loop
print("🤖 Yash's Resume Chatbot is ready! Type 'exit' to quit.\n")
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        print("Chatbot: Goodbye!")
        break
    response = qa_chain.run(query)
    print("Chatbot:", response)
