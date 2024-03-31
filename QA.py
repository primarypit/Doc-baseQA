from DataPreprocess import *
from VecRetrieval import *
from LLMAPI import *
from langchain import OpenAI
import warnings

warnings.filterwarnings('ignore')

class ConversationSession:
    def __init__(self, file_folder_dir, file_name, saving_dir, search_type, search_kwargs):
        self.doc = Document(file_folder_dir, file_name)
        self.doc_name = self.doc.get_name()
        self.summary = load_summary(self.doc_name)
        self.vecs = VecStore(search_type, search_kwargs)
        self.vecs.load_retriever(saving_dir, self.doc_name)
        self.llmapi = None
        self.history = []

    def set_api_key(self, api_key):
        self.llmapi = LLMapi(OpenAI(temperature=0, openai_api_key=api_key))

    def query_search(self, query):
        query_pre = ""
        for h in self.history:
            query_pre = query_pre + "user: " + h[0] + "\n" + "assistant: " + h[1] + "\n\n"
        query_pre.join('\n\n').join('The above is the history of the conversation\n\n')
        related_contents = self.vecs.query_search(query_pre + query)
        query = query_pre + "based on the above history and given related text, " + query
        related_contents = sorted(related_contents, key=lambda x: x.metadata["order"])
        related_text = ""
        related_text = related_text + "\n\n".join([r.page_content for r in related_contents])
        response = self.llmapi.get_response(query, self.summary, related_text)
        self.history.append((query, response))
        if len(self.history) > 30:
            self.history.pop(0)
        return response