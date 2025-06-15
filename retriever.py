# Allows agents to query the databse and use the data for promts
# add more specific retrival function here later
from langchain_ollama import OllamaEmbeddings
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
import os

embeddings = OllamaEmbeddings(model = "mxbai-embed-large")

db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

vectorstore = Chroma(
    collection_name="Lab_IOT",
    embedding_function=embeddings,
    persist_directory= db_location
)

retriever = vectorstore.as_retriever()

query = "What temperature was the bed around the beginning of the print?"
docs = retriever.get_relevant_documents(query)

for i, doc in enumerate(docs):
    print(f"\n--- Chunk {i + 1} ---")
    print(doc.page_content)
