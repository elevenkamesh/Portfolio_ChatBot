from langchain_community.document_loaders import PyPDFLoader, PlaywrightURLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.docstore.document import Document

# ---- Load PDF ----
loader = PyPDFLoader(r"C:\Users\kamesh\Documents\git\Portfolio_ChatBot\palani_kameshwaran.pdf")
pages = loader.load()

# ---- Load Website ----
urls = ["https://kameshx.vercel.app/"]
webloader = PlaywrightURLLoader(urls=urls, remove_selectors=["script", "style"])
webdocs = webloader.load()

# ---- Split into chunks ----
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
pdf_docs = splitter.split_documents(pages)
web_docs = splitter.split_documents(webdocs)

for d in pdf_docs: d.metadata["source"] = "pdf"
for d in web_docs: d.metadata["source"] = "website"


# ---- Combine all documents ----
all_docs = pdf_docs + web_docs

# ---- Convert to embeddings + store ----
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

db = Chroma.from_documents(all_docs, embeddings, persist_directory="./chroma_db")
db.persist()

print(f"✅ Stored {len(all_docs)} documents in Chroma DB")
