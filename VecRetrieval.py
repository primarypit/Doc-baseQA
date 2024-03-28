from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_community.vectorstores.faiss import FAISS
from faiss import IndexFlatL2
from langchain_community.docstore.in_memory import InMemoryDocstore
from DataPreprocess import Document
from tqdm import tqdm


class VecStore():
    
    def __init__(self, search_type: str, search_kwargs: dict):
        
        self.eb = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
        self.search_type = search_type
        self.search_kwargs = search_kwargs
        self.retriever = None

    def build_retriever(self, doc: Document, saveflag):
        doc_chunks = doc.text_split()
        print("Building retriever...")
        metadata = []
        for i in range(len(doc_chunks)):
            tmp = {}
            tmp["order"] = i
            metadata.append(tmp)
        db = FAISS.from_texts(doc_chunks, self.eb, metadata)
        if saveflag:
            db.save_local("FAISS_store", doc.get_name())
        #db = FAISS(self.eb, index=IndexFlatL2(1536), docstore=InMemoryDocstore(), index_to_docstore_id={})
        #pbar = tqdm(list(range(len(doc_chunks))),desc="Building vec store...")
        #for i in pbar:
            #db.add_texts([doc_chunks[i]])
        self.retriever = db.as_retriever(search_type = self.search_type, search_kwargs = self.search_kwargs)
    
    def load_retriever(self, folder_path, doc_name):

        db = FAISS.load_local(folder_path, self.eb, doc_name, allow_dangerous_deserialization = True)
        self.retriever = db.as_retriever(search_type = self.search_type, search_kwargs = self.search_kwargs)
    
    def query_search(self, query):
        docs = self.retriever.get_relevant_documents(query)
        return docs 