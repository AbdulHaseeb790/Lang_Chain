from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS

load_dotenv(dotenv_path=r"C:\Users\SOHAIL\phase7-llm-apis\.env")

embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")
vector_store = FAISS.load_local("faiss_index", embedding_model, allow_dangerous_deserialization=True)

query = "What is a network?"
results = vector_store.similarity_search(query, k=3)

for i, doc in enumerate(results):
    print(f"--- Chunk {i+1} ---")
    print(doc.page_content)
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

context = "\n\n".join([doc.page_content for doc in results])

prompt = ChatPromptTemplate.from_template(
    "Answer the question based only on the context below.\n\nContext: {context}\n\nQuestion: {question}"
)

llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
parser = StrOutputParser()

chain = prompt | llm | parser
answer = chain.invoke({"context": context, "question": query})

print(answer)