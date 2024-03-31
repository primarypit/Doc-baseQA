import time
from langchain.chains import LLMChain
from langchain_core.prompts import PromptTemplate

class LLMapi():
    def __init__(self, llm):
        self.llm = llm
        self.fiexed_prompt = ('Please Answer the above question given the following infomation of the text.')

    def get_response(self, query, summary, text):
        template = """Question: {query}
                    Summary of the given text/data: {summary}
                    Related text of the given text/data: {text}
                    Answer: """

        prompt = PromptTemplate(template=template, input_variables=["query", "summary", "text"])

        while True:
            try:
                llm_chain = LLMChain(prompt = prompt, llm = self.llm)
                ans = llm_chain.run(query = query, summary = summary, text = text)
                return ans
            except Exception as e:
                time.sleep(2)
                continue