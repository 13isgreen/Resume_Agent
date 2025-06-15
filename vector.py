from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_core.documents import Doccument
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.docstore.document import Document
from dotenv import load_dotenv
import os
import pandas as pd


# Read in data (possibly with pandas)
# Example of using pandas to read in a csv file
#df = pd.read_csv("example.csv")
# Feed all the manuals and data here

with open("serial.log", "r", encoding="utf-8") as f: # UTF-8 Neccissary for @ in logs
    raw_logs = f.read()

# Chunk up the Logs
# Write up something later thats more advanced and chucks by lines 
# 15 min log file had 2,445,051 chars
splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=50
)
log_chunks = splitter.split_text(raw_logs)

# Convert to LangChain Document format
documents = [Document(page_content=chunk) for chunk in log_chunks]

embeddings = OllamaEmbeddings(model = "mxbai-embed-large")

# Check to see of database exists else create
db_location = "./chroma_langchain_db"
add_documents = not os.path.exists(db_location)

# This is for parsing rows of the csv file example above and storing them
#if add_documents:
#    documents = []
#    ids = []
#
#    for i, row in df.iterrows():
#        document = Document(
#            page_content=row["Title"] + " " + row["Data"],
#            metadata = {"maintenance": row["maintenance"],"date": row["Date"]},   # Additional dimensions for vector database that are unqueryable
#            id = str(i)
#        )
#        ids.append(str(i))
#        documents.append(document)

vector_store = Chroma(
    collection_name = "LAB_IOT",
    persist_directory = db_location,
    embedding_function = embeddings
)

if add_documents:
    vector_store.add_documents(documents = documents, ids=ids)

# Import this retriever method in agents
# For more specific searches increase the (key word arguments)
retriever = vector_store.as_retriever(
    search_kwargs = {"k":5}
)