from DataPreprocess import *
from VecRetrieval import *
from LLMAPI import *
from langchain import OpenAI
import warnings
warnings.filterwarnings('ignore')

open_key="****"
llmai = OpenAI(temperature=0, openai_api_key=open_key)

file_folder_dir = "Docs/"
saving_dir = "FAISS_store/"
file_name = "Black_Cat.pdf"

#doc = Document(file_folder_dir, file_name)
#doc.summary(llmai)
#doc_name = doc.get_name()
doc_name = "Black_Cat"
summary = load_summary(doc_name)

search_type = "similarity"
search_kwargs = {"k": 10}

vecs = VecStore(search_type, search_kwargs)
#vecs.build_retriever(doc, llmai, True)
vecs.load_retriever(saving_dir, doc_name)

test_query = "How many cats appeared in the story"

related_contents = vecs.query_search(test_query)
related_contents = sorted(related_contents, key = lambda x: x.metadata["order"])

related_text = "\n\n".join([r.page_content for r in related_contents])
#print(related_text)

llmapi = LLMapi(llmai)
ans = llmapi.get_response(test_query, summary, related_text)
print(ans)