from QA import *

session = ConversationSession("Docs/", "Black_Cat.pdf", "FAISS_store/", "similarity", {"k": 10})

print("input api key:")
api_key = input()
session.set_api_key(api_key)

conversationCnt = 0
while True:
    conversationCnt += 1
    print("Conversation", conversationCnt)
    print("Query:")
    query = input()
    response = session.query_search(query)
    print("response:")
    print(response)