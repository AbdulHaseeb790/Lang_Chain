from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
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

texts = []
for chunk in chunks:
    texts.append(chunk.page_content)

vectors = embedding_model.embed_documents(texts)

print(len(vectors))
print(len(vectors[0]))