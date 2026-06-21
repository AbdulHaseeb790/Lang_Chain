from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community .vectorstores import FAISS
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv(dotenv_path=r'C:\Users\SOHAIL\phase7-llm-apis\.env')
loader=PyPDFLoader(r"C:\Users\SOHAIL\phase7-llm-apis\lang_chain\Rag\CN - Lect 01.pdf")
docs=loader.load()
print("len of docs",len(docs))
# print(docs[0].page_content[:200])
splitter=RecursiveCharacterTextSplitter(chunk_size=500,chunk_overlap=50)
chunks=splitter.split_documents(docs)
print("chunnks",len(chunks))
print("first chunks",chunks[0].page_content)
embedding_model=GoogleGenerativeAIEmbeddings(model='gemini-embedding-001')
vector_store=FAISS.from_documents(chunks,embedding_model)
print("vector store created")
vector_store.save_local("faiss_index")
print("SAVED TO DISK")