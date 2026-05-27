import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI 

load_dotenv()

class LLM:
    def __init__(self):
        
        # 从 .env 中读取配置，后面的是默认值（Fallback）
        self.model = os.getenv('LLM_MODEL', 'gpt-4o')
        self.base_url = os.getenv('LLM_BASE_URL', 'https://api.openai.com/v1')
        self.api_key = os.getenv('LLM_API_KEY')
        
        # 初始化 ChatOpenAI
        self.llm = ChatOpenAI(
            model=self.model,
            base_url=self.base_url,
            api_key=self.api_key,
            verbose=True
        )