from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

VECTORSTORE_PATH = "faiss_index"


# def build_vectorstore_from_pdf(pdf_path: str):
#     loader = PyPDFLoader(pdf_path)
#     docs = loader.load()
def build_vectorstore_from_pdf(pdf_path: str):

    import os

    print("RAW =", repr(pdf_path))
    print("EXISTS =", os.path.exists(pdf_path))
    print("ISFILE =", os.path.isfile(pdf_path))

    loader = PyPDFLoader(pdf_path)
    docs = loader.load()

    
def load_vectorstore():
    return FAISS.load_local(
        VECTORSTORE_PATH,
        embeddings,
        allow_dangerous_deserialization=True
    )


def retrieve_docs(query: str):
    db = load_vectorstore()

    retriever = db.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4}
    )

    docs = retriever.invoke(query)

    return "\n\n".join(d.page_content for d in docs)