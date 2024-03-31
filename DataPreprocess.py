from langchain_community.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter,RecursiveCharacterTextSplitter
from langchain.chains.summarize import load_summarize_chain

def load_summary(doc_name):
    with open("Summary/{}_summary.txt".format(doc_name), "r", encoding = "utf-8") as f:
        summary = f.read()
    return summary

class Document():

    def __init__(self, folder_dir, file_name):
        self.file_dir = folder_dir + file_name
        self.doc_name = file_name.split(".")[0]
        self.type = file_name.split(".")[-1]
        self.docs = self.load_text(self.file_dir)
    
    def load_text(self,file_dir):
        if self.type == "pdf":
            loader = UnstructuredFileLoader(file_dir, mode="elements")
        elif self.type == "txt":
            loader =  UnstructuredFileLoader(file_dir)

        print("Loading...")
        docs = loader.load()

        return docs
    
    def summary(self, llm):
        print("Summarizing...")
        docs = [doc.page_content for doc in self.docs]

        text = ""
        for doc in docs:
            text += doc
            text += " "

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", ".", "!", "\"", " "],
            keep_separator = False,
            chunk_size = 10000,
            chunk_overlap  = 500,
            length_function = len,
        )

        summary_chunks = text_splitter.create_documents([text])

        chain = load_summarize_chain(llm = llm, chain_type = 'map_reduce',  verbose = False) 
        summary = chain.run(summary_chunks)
        with open("Summary/{}_summary.txt".format(self.doc_name), "w", encoding = "utf-8") as f:
            f.write(summary)

    def text_split(self):
        print("Spiltting...")

        text_splitter = RecursiveCharacterTextSplitter(
            separators=["\n\n", ".", "!", "\"", " "],
            keep_separator = False,
            chunk_size = 196,
            chunk_overlap  = 0.2,
            length_function = len,
        )

        chunks = text_splitter.split_documents(self.docs)
        return chunks

    def get_name(self):
        return self.doc_name