from langchain_community.document_loaders  import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings


# extract the data from pdf 
loader = PyPDFLoader("C:\Users\kamesh\Documents\git\Portfolio_ChatBot\palani_kameshwaran.pdf")
pages = loader.load()



# spilt into chuncks 

splitter = RecursiveCharacterTextSplitter(chuck_size=500 , chuck_overlap=50)
docs = splitter.spilt_documents(pages)

# Convert to embeddings + store

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma.from_documents(docs, embeddings, persist_directory="./chroma_db")