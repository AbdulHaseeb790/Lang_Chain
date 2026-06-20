from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

load_dotenv("../.env")

loader = TextLoader('Data/sample.txt')
document = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)

chunks = splitter.split_documents(document)

embedding_model = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

vector_store = FAISS.from_documents(chunks, embedding_model)

query = "What is an AI agent?"
results = vector_store.similarity_search(query, k=2)

for r in results:
    print(r.page_content)
    print("---")