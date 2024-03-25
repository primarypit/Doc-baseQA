from DataPreprocess import *
from VecRetrieval import *
from LLMAPI import *
import warnings
warnings.filterwarnings('ignore')

k = 5

doc = Document("Docs/Black_Cat.pdf")

db = VecStore(doc)

test_query = "What did I do to the cat"

related_contents = db.query_search(test_query, k)
related_text = "\n\n".join([r.page_content for r in related_contents])

llmapi = LLMapi()
print(llmapi.get_response(test_query, related_text))