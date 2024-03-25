import time
from openai import OpenAI

class LLMapi():
    def __init__(self):

        self.client = OpenAI(
            api_key="api-key",
            base_url="*************"
        )
        self.model = "gpt-3.5-turbo"
        self.fiexed_prompt = ('Please Answer the above question given the following text.')

    def get_response(self, query, text):
        prompt = "\n".join([query, self.fiexed_prompt, text])
        while True:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": "You are a helpful assistant."},
                        {"role": "user", "content": prompt}
                    ]
                )
                return response.choices[0].message.content
            except:
                time.sleep(2)
                continue