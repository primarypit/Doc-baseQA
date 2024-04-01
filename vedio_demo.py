from DataPreprocess import *
from VecRetrieval import *
from LLMAPI import *
from langchain import OpenAI
import warnings
import re
warnings.filterwarnings('ignore')

open_key="***"
llmai = OpenAI(temperature=0, openai_api_key=open_key)

saving_dir = "FAISS_store/"
url = "https://www.youtube.com/watch?v=zIwLWfaAg-8"

video = Video(url)
video.summary(llmai)
video_name = video.get_name()
summary = load_summary(video_name)

search_type = "similarity"
search_kwargs = {"k": 10}

vecs = VecStore(search_type, search_kwargs)
vecs.build_retriever(video, True)
vecs.load_retriever(saving_dir, video_name)

test_query = "What kind of future does Elon Musk mention"

related_contents = vecs.query_search(test_query)
related_contents = sorted(related_contents, key = lambda x: x.metadata["order"])

related_text = "\n\n".join([r.page_content for r in related_contents])
#print(related_text)

llmapi = LLMapi(llmai)
ans = llmapi.get_response(test_query, summary, related_text)
print(ans)