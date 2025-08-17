import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAI
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
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

# Custom prompt template
# Custom prompt template (mention uploaded .txt file)
template = """
You are Yash's Resume Chatbot.
The context comes from an uploaded text file called 'yash_resume.txt', which contains Yash's resume.
not used my , I , me, my, or mine. proide answer in yash name
If asked about education, answer only based on the content of the uploaded file.
everytime complete full sentanece if limit exist stop before the limit.

Context from yash_resume.txt:
{context}

Question: {question}
Answer:
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=template
)

# Create QA chain with custom prompt
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(search_kwargs={"k": 6}),
    chain_type="stuff", #means use the full context 
    chain_type_kwargs={"prompt": prompt}
)

# Chat loop
print("🤖 Yash's Resume Chatbot is ready! Type 'exit' to quit.\n")
while True:
    query = input("You: ")
    if query.lower() in ["exit", "quit"]:
        print("Chatbot: Goodbye!")
        break
    response = qa_chain.run(query)
    print("Chatbot:", response)
