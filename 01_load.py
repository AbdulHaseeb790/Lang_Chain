from langchain_community.document_loaders import TextLoader
loader=TextLoader('Data/sample.txt')
document=loader.load()
print(document[0].metadata)