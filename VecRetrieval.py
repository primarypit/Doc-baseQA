from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain_community.vectorstores import DocArrayInMemorySearch
from DataPreprocess import Document
from tqdm import tqdm
# Deep Lake, FAISS

class VecStore():
    
    def __init__(self, doc: Document):
        
        self.eb = HuggingFaceBgeEmbeddings(model_name="BAAI/bge-large-zh-v1.5")
        self.doc = doc
        self.database = self.build_database()

    def build_database(self):
        doc_chunks = self.doc.text_split()
        db = DocArrayInMemorySearch.from_params(self.eb) #initialize
        pbar = tqdm(list(range(len(doc_chunks))),desc="Building vec store...")
        for i in pbar:
            db.add_texts([doc_chunks[i]])
        return db
    
    def query_search(self, query, k):
        docs = self.database.similarity_search(query, k)
        return docs 