from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
loader=TextLoader("Data/sample.txt")
document=loader.load()
splitter=RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=20
)
chunks=splitter.split_documents(document)
print(len(chunks))
print(chunks[0].page_content)
print("---")
print(chunks[1].page_content)
for i, chunk in enumerate(chunks):
    print(f"--- Chunk {i} ---")
    print(chunk.page_content)
    print()