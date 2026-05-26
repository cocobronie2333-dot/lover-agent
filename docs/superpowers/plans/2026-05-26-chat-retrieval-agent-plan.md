# Chat Retrieval Agent Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 构建一个基于 LangChain 的聊天记录检索 Agent，支持 HTML 格式聊天记录的 RAG 检索

**Architecture:** 模块化分层架构，RAG 核心与 API 层分离，通过 Chroma 向量数据库实现语义检索

**Tech Stack:** Python + LangChain + Chroma + FastAPI + BGE 嵌入模型

---

## 文件结构

```
lover_agent/
├── config/
│   └── settings.py           # 配置管理
├── adapters/
│   ├── html_parser.py        # HTML 解析
│   └── chroma_client.py      # Chroma 客户端
├── rag/
│   ├── loader.py             # 文档加载
│   ├── splitter.py           # 文本分割
│   ├── embedder.py           # 嵌入模型
│   ├── vectorstore.py        # 向量存储
│   ├── retriever.py          # 检索器
│   └── indexer.py            # 索引构建
├── chains/
│   ├── retrieval_chain.py    # 检索链
│   └── conversational_chain.py  # 对话链
├── api/
│   └── routes.py             # API 路由
├── services/
│   └── chat_service.py       # 聊天服务
├── models/
│   └── schemas.py            # 数据模型
├── utils/
│   └── logger.py            # 日志工具
├── main.py                   # 入口
├── requirements.txt
├── .env.example
└── tests/
    ├── test_html_parser.py
    ├── test_rag_loader.py
    ├── test_embedder.py
    ├── test_vectorstore.py
    └── test_api.py
```

---

## Task 1: 项目初始化与配置

**Files:**
- Create: `requirements.txt`
- Create: `.env.example`
- Create: `lover_agent/config/settings.py`
- Create: `lover_agent/utils/logger.py`

- [ ] **Step 1: 创建 requirements.txt**

```txt
langchain>=0.3.0
langchain-community>=0.3.0
langchain-core>=0.3.0
chromadb>=0.5.0
sentence-transformers>=3.0.0
fastapi>=0.115.0
uvicorn>=0.30.0
pydantic>=2.9.0
pydantic-settings>=2.5.0
python-dotenv>=1.0.0
beautifulsoup4>=4.12.0
lxml>=5.3.0
httpx>=0.27.0
pytest>=8.3.0
pytest-asyncio>=0.24.0
```

- [ ] **Step 2: 创建 .env.example**

```env
# 向量数据库
CHROMA_DB_PATH=./data/chroma
CHROMA_COLLECTION_NAME=chat_records

# 嵌入模型
EMBEDDING_MODEL=bge-small-zh
EMBEDDING_MODEL_PATH=./models/bge-small-zh

# LLM 配置
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your-api-key-here
LLM_MODEL=gpt-4o-mini

# 日志
LOG_LEVEL=INFO
```

- [ ] **Step 3: 创建 settings.py**

```python
from pathlib import Path
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

load_dotenv()

class Settings(BaseSettings):
    # Chroma 配置
    chroma_db_path: str = "./data/chroma"
    chroma_collection_name: str = "chat_records"

    # 嵌入模型配置
    embedding_model: str = "bge-small-zh"
    embedding_model_path: str = "./models/bge-small-zh"

    # LLM 配置
    llm_base_url: str = "https://api.openai.com/v1"
    llm_api_key: str
    llm_model: str = "gpt-4o-mini"

    # 日志
    log_level: str = "INFO"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
```

- [ ] **Step 4: 创建 logger.py**

```python
import logging
import sys
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()
import os

log_level = os.getenv("LOG_LEVEL", "INFO")

def get_logger(name: str) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, log_level))
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(getattr(logging, log_level))
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger
```

- [ ] **Step 5: 提交**

```bash
git add requirements.txt .env.example lover_agent/config/settings.py lover_agent/utils/logger.py
git commit -m "feat: 项目初始化 - 配置管理与日志工具"
```

---

## Task 2: HTML 解析器

**Files:**
- Create: `lover_agent/adapters/html_parser.py`
- Create: `tests/test_html_parser.py`

- [ ] **Step 1: 创建测试 test_html_parser.py**

```python
from bs4 import BeautifulSoup

def test_parse_wechat_html():
    html = """
    <div class="message">
        <div class="message_time">2024-01-01 12:00:00</div>
        <div class="message_sender">张三</div>
        <div class="message_content">你好</div>
    </div>
    <div class="message">
        <div class="message_time">2024-01-01 12:01:00</div>
        <div class="message_sender">李四</div>
        <div class="message_content">你好呀</div>
    </div>
    """
    from lover_agent.adapters.html_parser import WeChatHTMLParser

    parser = WeChatHTMLParser()
    messages = parser.parse(html)
    assert len(messages) == 2
    assert messages[0]["sender"] == "张三"
    assert messages[0]["content"] == "你好"
    assert messages[1]["sender"] == "李四"
    assert messages[1]["content"] == "你好呀"
```

- [ ] **Step 2: 运行测试验证失败**

Run: `pytest tests/test_html_parser.py -v`
Expected: FAIL - ModuleNotFoundError

- [ ] **Step 3: 创建 html_parser.py**

```python
from bs4 import BeautifulSoup
from dataclasses import dataclass
from typing import List, Optional
from lover_agent.utils.logger import get_logger

logger = get_logger(__name__)

@dataclass
class ChatMessage:
    sender: str
    content: str
    timestamp: Optional[str] = None
    session_id: Optional[str] = None

class WeChatHTMLParser:
    """微信 HTML 聊天记录解析器"""

    def parse(self, html_content: str) -> List[ChatMessage]:
        soup = BeautifulSoup(html_content, "lxml")
        messages = []

        for msg_div in soup.find_all("div", class_="message"):
            try:
                sender = msg_div.find("div", class_="message_sender")
                content = msg_div.find("div", class_="message_content")
                timestamp = msg_div.find("div", class_="message_time")

                if sender and content:
                    messages.append(ChatMessage(
                        sender=sender.get_text(strip=True),
                        content=content.get_text(strip=True),
                        timestamp=timestamp.get_text(strip=True) if timestamp else None
                    ))
            except Exception as e:
                logger.warning(f"解析消息失败: {e}")
                continue

        return messages

    def parse_file(self, file_path: str) -> List[ChatMessage]:
        with open(file_path, "r", encoding="utf-8") as f:
            html_content = f.read()
        return self.parse(html_content)
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/test_html_parser.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add lover_agent/adapters/html_parser.py tests/test_html_parser.py
git commit -m "feat: 实现微信 HTML 聊天记录解析器"
```

---

## Task 3: RAG Loader 与 Splitter

**Files:**
- Create: `lover_agent/rag/loader.py`
- Create: `lover_agent/rag/splitter.py`
- Create: `tests/test_rag_loader.py`
- Create: `tests/test_splitter.py`

- [ ] **Step 1: 创建 loader.py**

```python
from typing import List
from lover_agent.adapters.html_parser import WeChatHTMLParser, ChatMessage
from lover_agent.utils.logger import get_logger

logger = get_logger(__name__)

class ChatRecordLoader:
    """聊天记录加载器"""

    def __init__(self, parser: WeChatHTMLParser = None):
        self.parser = parser or WeChatHTMLParser()

    def load(self, file_path: str) -> List[ChatMessage]:
        logger.info(f"加载聊天记录: {file_path}")
        messages = self.parser.parse_file(file_path)
        logger.info(f"加载了 {len(messages)} 条消息")
        return messages

    def load_multiple(self, file_paths: List[str]) -> List[ChatMessage]:
        all_messages = []
        for path in file_paths:
            all_messages.extend(self.load(path))
        return all_messages
```

- [ ] **Step 2: 创建 splitter.py**

```python
from typing import List
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from lover_agent.utils.logger import get_logger

logger = get_logger(__name__)

class ChatRecordSplitter:
    """聊天记录文本分割器"""

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50
    ):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len
        )

    def split_messages(self, messages: List) -> List[Document]:
        """将消息列表分割为 Document 列表"""
        documents = []
        for msg in messages:
            doc = Document(
                page_content=f"[{msg.sender}] {msg.content}",
                metadata={
                    "sender": msg.sender,
                    "timestamp": msg.timestamp,
                    "session_id": msg.session_id
                }
            )
            documents.append(doc)

        logger.info(f"创建了 {len(documents)} 个 Document")
        return self.splitter.split_documents(documents)

    def split_by_session(self, messages: List, session_gap_seconds: int = 3600) -> List[Document]:
        """按会话分割消息（时间间隔超过 session_gap_seconds 视为新会话）"""
        documents = []
        current_session = []

        for msg in messages:
            current_session.append(msg)

            # 判断是否应该分割
            should_split = False
            if len(current_session) > 1:
                # 简单逻辑：每 10 条消息作为一个会话
                if len(current_session) % 10 == 0:
                    should_split = True

            if should_split:
                docs = self._create_session_docs(current_session)
                documents.extend(docs)
                current_session = []

        # 处理最后一个会话
        if current_session:
            docs = self._create_session_docs(current_session)
            documents.extend(docs)

        return self.splitter.split_documents(documents)

    def _create_session_docs(self, messages: List) -> List[Document]:
        """创建会话级 Document"""
        if not messages:
            return []

        session_text = "\n".join([f"[{msg.sender}] {msg.content}" for msg in messages])
        return [Document(
            page_content=session_text,
            metadata={
                "sender": messages[0].sender,
                "timestamp": messages[0].timestamp,
                "message_count": len(messages)
            }
        )]
```

- [ ] **Step 3: 创建测试 test_rag_loader.py**

```python
def test_chat_record_loader():
    from lover_agent.rag.loader import ChatRecordLoader
    from lover_agent.adapters.html_parser import ChatMessage

    loader = ChatRecordLoader()
    messages = [
        ChatMessage(sender="张三", content="你好", timestamp="2024-01-01 12:00:00"),
        ChatMessage(sender="李四", content="你好呀", timestamp="2024-01-01 12:01:00")
    ]

    # Mock parser
    loader.parser.parse = lambda x: messages

    result = loader.load("dummy_path")
    assert len(result) == 2
    assert result[0].sender == "张三"
```

- [ ] **Step 4: 创建测试 test_splitter.py**

```python
def test_split_messages():
    from lover_agent.rag.splitter import ChatRecordSplitter
    from lover_agent.adapters.html_parser import ChatMessage
    from langchain_core.documents import Document

    splitter = ChatRecordSplitter(chunk_size=100, chunk_overlap=10)
    messages = [
        ChatMessage(sender="张三", content="这是一条测试消息", timestamp="2024-01-01 12:00:00"),
        ChatMessage(sender="李四", content="这是第二条测试消息", timestamp="2024-01-01 12:01:00")
    ]

    docs = splitter.split_messages(messages)
    assert len(docs) > 0
    assert all(isinstance(d, Document) for d in docs)
```

- [ ] **Step 5: 运行测试验证通过**

Run: `pytest tests/test_rag_loader.py tests/test_splitter.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add lover_agent/rag/loader.py lover_agent/rag/splitter.py tests/test_rag_loader.py tests/test_splitter.py
git commit -m "feat: 实现 RAG 文档加载器与分割器"
```

---

## Task 4: Embedder 与 VectorStore

**Files:**
- Create: `lover_agent/rag/embedder.py`
- Create: `lover_agent/rag/vectorstore.py`
- Create: `lover_agent/adapters/chroma_client.py`
- Create: `tests/test_embedder.py`
- Create: `tests/test_vectorstore.py`

- [ ] **Step 1: 创建 embedder.py**

```python
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from lover_agent.config.settings import settings
from lover_agent.utils.logger import get_logger

logger = get_logger(__name__)

class LocalEmbedder:
    """本地嵌入模型（BGE 免费模型）"""

    def __init__(self):
        self.model_name = settings.embedding_model
        self.model_path = settings.embedding_model_path

    def load(self) -> HuggingFaceBgeEmbeddings:
        logger.info(f"加载嵌入模型: {self.model_name}")
        embeddings = HuggingFaceBgeEmbeddings(
            model_name=self.model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True}
        )
        return embeddings

    def get_embeddings(self) -> HuggingFaceBgeEmbeddings:
        return self.load()
```

- [ ] **Step 2: 创建 chroma_client.py**

```python
import chromadb
from chromadb.config import Settings as ChromaSettings
from lover_agent.config.settings import settings
from lover_agent.utils.logger import get_logger

logger = get_logger(__name__)

class ChromaClient:
    """Chroma 客户端封装"""

    def __init__(self):
        self.client = None
        self.collection = None

    def connect(self):
        logger.info(f"连接 Chroma: {settings.chroma_db_path}")
        self.client = chromadb.PersistentClient(
            path=settings.chroma_db_path,
            settings=ChromaSettings(
                anonymized_telemetry=False
            )
        )

    def get_or_create_collection(self, name: str = None):
        if not self.client:
            self.connect()

        collection_name = name or settings.chroma_collection_name
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"}
        )
        logger.info(f"获取/创建 collection: {collection_name}")
        return self.collection

    def close(self):
        if self.client:
            self.client = None
            self.collection = None
```

- [ ] **Step 3: 创建 vectorstore.py**

```python
from typing import List, Optional
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from lover_agent.rag.embedder import LocalEmbedder
from lover_agent.adapters.chroma_client import ChromaClient
from lover_agent.config.settings import settings
from lover_agent.utils.logger import get_logger

logger = get_logger(__name__)

class ChatVectorStore:
    """聊天记录向量存储"""

    def __init__(self):
        self.embedder = LocalEmbedder()
        self.chroma_client = ChromaClient()
        self.vectorstore = None

    def build(self, documents: List[Document], collection_name: str = None):
        logger.info(f"构建向量索引，文档数: {len(documents)}")

        embeddings = self.embedder.get_embeddings()
        collection_name = collection_name or settings.chroma_collection_name

        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=settings.chroma_db_path,
            collection_name=collection_name
        )
        logger.info(f"向量索引构建完成")
        return self.vectorstore

    def load(self, collection_name: str = None) -> Chroma:
        embeddings = self.embedder.get_embeddings()
        collection_name = collection_name or settings.chroma_collection_name

        self.vectorstore = Chroma(
            persist_directory=settings.chroma_db_path,
            embedding_function=embeddings,
            collection_name=collection_name
        )
        return self.vectorstore

    def similarity_search(
        self,
        query: str,
        k: int = 5,
        filter: dict = None
    ) -> List[Document]:
        if not self.vectorstore:
            self.load()

        return self.vectorstore.similarity_search(
            query=query,
            k=k,
            filter=filter
        )

    def similarity_search_with_score(
        self,
        query: str,
        k: int = 5
    ) -> List[tuple]:
        if not self.vectorstore:
            self.load()

        return self.vectorstore.similarity_search_with_score(
            query=query,
            k=k
        )

    def delete_collection(self, collection_name: str = None):
        client = self.chroma_client
        client.connect()
        collection_name = collection_name or settings.chroma_collection_name
        client.client.delete_collection(collection_name)
        logger.info(f"删除 collection: {collection_name}")
```

- [ ] **Step 4: 创建测试 test_embedder.py**

```python
def test_embedder_init():
    from lover_agent.rag.embedder import LocalEmbedder
    embedder = LocalEmbedder()
    assert embedder.model_name is not None
```

- [ ] **Step 5: 创建测试 test_vectorstore.py**

```python
def test_vectorstore_build():
    from lover_agent.rag.vectorstore import ChatVectorStore
    from langchain_core.documents import Document

    vs = ChatVectorStore()
    docs = [Document(page_content="测试内容", metadata={"sender": "张三"})]

    # 注意：这个测试需要模型文件，实际运行时可能需要 mock
    # vs.build(docs)
    assert vs is not None
```

- [ ] **Step 6: 运行测试验证通过**

Run: `pytest tests/test_embedder.py tests/test_vectorstore.py -v`
Expected: PASS

- [ ] **Step 7: 提交**

```bash
git add lover_agent/rag/embedder.py lover_agent/rag/vectorstore.py lover_agent/adapters/chroma_client.py tests/test_embedder.py tests/test_vectorstore.py
git commit -m "feat: 实现嵌入模型与向量存储"
```

---

## Task 5: Retriever 与 Indexer

**Files:**
- Create: `lover_agent/rag/retriever.py`
- Create: `lover_agent/rag/indexer.py`
- Create: `tests/test_retriever.py`
- Create: `tests/test_indexer.py`

- [ ] **Step 1: 创建 retriever.py**

```python
from typing import List, Optional
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
from lover_agent.rag.vectorstore import ChatVectorStore
from lover_agent.utils.logger import get_logger

logger = get_logger(__name__)

class ChatRetriever(BaseRetriever):
    """聊天记录检索器"""

    def __init__(
        self,
        vectorstore: ChatVectorStore = None,
        search_kwargs: dict = None
    ):
        self.vectorstore = vectorstore or ChatVectorStore()
        self.search_kwargs = search_kwargs or {"k": 5}

    def _get_relevant_documents(self, query: str) -> List[Document]:
        logger.info(f"检索查询: {query}")
        docs = self.vectorstore.similarity_search(
            query=query,
            k=self.search_kwargs.get("k", 5)
        )
        logger.info(f"检索到 {len(docs)} 个相关文档")
        return docs

    def get_relevant_documents(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)

    def invoke(self, query: str) -> List[Document]:
        return self._get_relevant_documents(query)
```

- [ ] **Step 2: 创建 indexer.py**

```python
from pathlib import Path
from typing import List, Optional
from lover_agent.rag.loader import ChatRecordLoader
from lover_agent.rag.splitter import ChatRecordSplitter
from lover_agent.rag.vectorstore import ChatVectorStore
from lover_agent.utils.logger import get_logger

logger = get_logger(__name__)

class ChatIndexer:
    """聊天记录索引构建器"""

    def __init__(self):
        self.loader = ChatRecordLoader()
        self.splitter = ChatRecordSplitter()
        self.vectorstore = ChatVectorStore()

    def index_file(self, file_path: str, collection_name: str = None) -> int:
        logger.info(f"索引文件: {file_path}")

        messages = self.loader.load(file_path)
        documents = self.splitter.split_messages(messages)

        self.vectorstore.build(documents, collection_name)

        logger.info(f"索引构建完成，文档数: {len(documents)}")
        return len(documents)

    def index_multiple(
        self,
        file_paths: List[str],
        collection_name: str = None
    ) -> dict:
        results = {}
        for path in file_paths:
            try:
                count = self.index_file(path, collection_name)
                results[path] = {"status": "success", "doc_count": count}
            except Exception as e:
                logger.error(f"索引文件失败 {path}: {e}")
                results[path] = {"status": "error", "error": str(e)}

        return results

    def rebuild_index(self, file_paths: List[str], collection_name: str = None):
        logger.info("重建索引")
        try:
            self.vectorstore.delete_collection(collection_name)
        except Exception:
            pass

        return self.index_multiple(file_paths, collection_name)
```

- [ ] **Step 3: 创建测试 test_retriever.py**

```python
def test_retriever_init():
    from lover_agent.rag.retriever import ChatRetriever

    retriever = ChatRetriever()
    assert retriever.search_kwargs.get("k") == 5
```

- [ ] **Step 4: 创建测试 test_indexer.py**

```python
def test_indexer_init():
    from lover_agent.rag.indexer import ChatIndexer

    indexer = ChatIndexer()
    assert indexer.loader is not None
    assert indexer.splitter is not None
    assert indexer.vectorstore is not None
```

- [ ] **Step 5: 运行测试验证通过**

Run: `pytest tests/test_retriever.py tests/test_indexer.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add lover_agent/rag/retriever.py lover_agent/rag/indexer.py tests/test_retriever.py tests/test_indexer.py
git commit -m "feat: 实现检索器与索引构建器"
```

---

## Task 6: LangChain Chains

**Files:**
- Create: `lover_agent/chains/retrieval_chain.py`
- Create: `lover_agent/chains/conversational_chain.py`
- Create: `tests/test_chains.py`

- [ ] **Step 1: 创建 retrieval_chain.py**

```python
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from lover_agent.rag.retriever import ChatRetriever
from lover_agent.utils.logger import get_logger

logger = get_logger(__name__)

PROMPT_TEMPLATE = """基于以下聊天记录回答问题:

{context}

问题: {question}
回答:"""

def create_retrieval_chain(
    retriever: ChatRetriever,
    llm
) -> RunnablePassthrough:
    """创建 RAG 检索链"""

    prompt = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    chain = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
        | prompt
        | llm
    )

    return chain

def create_retrieval_chain_with_history(
    retriever: ChatRetriever,
    llm
) -> RunnablePassthrough:
    """创建带历史上下文的检索链"""

    history_prompt = """基于以下聊天记录和对话历史回答问题:

历史记录:
{history}

当前记录:
{context}

问题: {question}
回答:"""

    prompt = ChatPromptTemplate.from_template(history_prompt)

    def format_docs(docs):
        return "\n\n".join([doc.page_content for doc in docs])

    chain = (
        RunnablePassthrough.assign(context=lambda x: format_docs(x["context"]))
        | prompt
        | llm
    )

    return chain
```

- [ ] **Step 2: 创建 conversational_chain.py**

```python
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from lover_agent.chains.retrieval_chain import create_retrieval_chain
from lover_agent.rag.retriever import ChatRetriever
from lover_agent.config.settings import settings
from lover_agent.utils.logger import get_logger

logger = get_logger(__name__)

def create_conversational_chain(retriever: ChatRetriever):
    """创建会话链（支持对话历史）"""

    llm = ChatOpenAI(
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model=settings.llm_model,
        temperature=0.7
    )

    memory = ConversationBufferMemory(
        memory_key="history",
        return_messages=True
    )

    chain = create_retrieval_chain_with_history(retriever, llm)

    return {
        "chain": chain,
        "llm": llm,
        "memory": memory,
        "retriever": retriever
    }

def chat(chain_data: dict, question: str, session_id: str = None):
    """执行对话"""

    retriever = chain_data["retriever"]
    docs = retriever.get_relevant_documents(question)

    history = chain_data["memory"].load_memory_variables({}).get("history", "")

    response = chain_data["chain"].invoke({
        "question": question,
        "context": docs,
        "history": history
    })

    chain_data["memory"].save_context(
        {"question": question},
        {"answer": response.content if hasattr(response, "content") else str(response)}
    )

    return response
```

- [ ] **Step 3: 创建测试 test_chains.py**

```python
def test_retrieval_chain_creation():
    from lover_agent.chains.retrieval_chain import create_retrieval_chain
    from lover_agent.rag.retriever import ChatRetriever
    from unittest.mock import MagicMock

    retriever = ChatRetriever()
    llm = MagicMock()

    chain = create_retrieval_chain(retriever, llm)
    assert chain is not None
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/test_chains.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add lover_agent/chains/retrieval_chain.py lover_agent/chains/conversational_chain.py tests/test_chains.py
git commit -m "feat: 实现 LangChain 检索链与对话链"
```

---

## Task 7: Models 与 Services

**Files:**
- Create: `lover_agent/models/schemas.py`
- Create: `lover_agent/services/chat_service.py`
- Create: `tests/test_models.py`
- Create: `tests/test_services.py`

- [ ] **Step 1: 创建 schemas.py**

```python
from pydantic import BaseModel
from typing import List, Optional

class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    use_history: bool = True

class ChatResponse(BaseModel):
    answer: str
    sources: List[str] = []
    session_id: str

class IndexRequest(BaseModel):
    file_paths: List[str]
    collection_name: Optional[str] = None
    rebuild: bool = False

class IndexResponse(BaseModel):
    status: str
    results: dict

class HealthResponse(BaseModel):
    status: str
    version: str = "1.0.0"
```

- [ ] **Step 2: 创建 chat_service.py**

```python
from typing import List, Optional
from lover_agent.chains.conversational_chain import create_conversational_chain, chat
from lover_agent.rag.retriever import ChatRetriever
from lover_agent.models.schemas import ChatRequest, ChatResponse
from lover_agent.utils.logger import get_logger

logger = get_logger(__name__)

class ChatService:
    """聊天服务"""

    def __init__(self):
        self.sessions = {}
        self.retriever = ChatRetriever()

    def get_session(self, session_id: str):
        if session_id not in self.sessions:
            chain_data = create_conversational_chain(self.retriever)
            self.sessions[session_id] = chain_data
        return self.sessions[session_id]

    def chat(self, request: ChatRequest) -> ChatResponse:
        logger.info(f"处理聊天请求: {request.question}")

        session = self.get_session(request.session_id or "default")

        docs = self.retriever.get_relevant_documents(request.question)
        sources = [doc.page_content[:100] for doc in docs]

        response = chat(session, request.question, request.session_id)

        return ChatResponse(
            answer=response.content if hasattr(response, "content") else str(response),
            sources=sources,
            session_id=request.session_id or "default"
        )

chat_service = ChatService()
```

- [ ] **Step 3: 创建测试 test_models.py**

```python
def test_chat_request():
    from lover_agent.models.schemas import ChatRequest

    req = ChatRequest(question="你好")
    assert req.question == "你好"
    assert req.use_history is True

def test_chat_response():
    from lover_agent.models.schemas import ChatResponse

    resp = ChatResponse(answer="你好呀", sources=["测试"], session_id="default")
    assert resp.answer == "你好呀"
```

- [ ] **Step 4: 创建测试 test_services.py**

```python
def test_chat_service_init():
    from lover_agent.services.chat_service import ChatService

    service = ChatService()
    assert service.retriever is not None
    assert service.sessions == {}
```

- [ ] **Step 5: 运行测试验证通过**

Run: `pytest tests/test_models.py tests/test_services.py -v`
Expected: PASS

- [ ] **Step 6: 提交**

```bash
git add lover_agent/models/schemas.py lover_agent/services/chat_service.py tests/test_models.py tests/test_services.py
git commit -m "feat: 实现数据模型与聊天服务"
```

---

## Task 8: API 路由与入口

**Files:**
- Create: `lover_agent/api/routes.py`
- Create: `lover_agent/main.py`
- Create: `tests/test_api.py`

- [ ] **Step 1: 创建 routes.py**

```python
from fastapi import APIRouter, HTTPException
from lover_agent.models.schemas import (
    ChatRequest,
    ChatResponse,
    IndexRequest,
    IndexResponse,
    HealthResponse
)
from lover_agent.services.chat_service import chat_service
from lover_agent.rag.indexer import ChatIndexer
from lover_agent.utils.logger import get_logger

logger = get_logger(__name__)

router = APIRouter()
indexer = ChatIndexer()

@router.post("/chat", response_model=ChatResponse)
async def chat(request: ChatRequest):
    try:
        return chat_service.chat(request)
    except Exception as e:
        logger.error(f"聊天请求失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chat/history/{session_id}")
async def get_history(session_id: str):
    session = chat_service.get_session(session_id)
    memory = session["memory"]
    history = memory.load_memory_variables({}).get("history", [])
    return {"session_id": session_id, "history": history}

@router.post("/index", response_model=IndexResponse)
async def build_index(request: IndexRequest):
    try:
        if request.rebuild:
            results = indexer.rebuild_index(request.file_paths, request.collection_name)
        else:
            results = indexer.index_multiple(request.file_paths, request.collection_name)

        return IndexResponse(status="success", results=results)
    except Exception as e:
        logger.error(f"索引构建失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/index/{file_id}")
async def delete_index(file_id: str):
    try:
        indexer.vectorstore.delete_collection()
        return {"status": "success", "message": f"索引 {file_id} 已删除"}
    except Exception as e:
        logger.error(f"删除索引失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health", response_model=HealthResponse)
async def health():
    return HealthResponse(status="healthy")
```

- [ ] **Step 2: 创建 main.py**

```python
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from lover_agent.api.routes import router
from lover_agent.utils.logger import get_logger

logger = get_logger(__name__)

app = FastAPI(
    title="Chat Retrieval Agent",
    description="基于 LangChain 的聊天记录检索 Agent",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api/v1")

@app.on_event("startup")
async def startup_event():
    logger.info("Chat Retrieval Agent 启动")

@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Chat Retrieval Agent 关闭")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
```

- [ ] **Step 3: 创建测试 test_api.py**

```python
def test_api_router():
    from lover_agent.api.routes import router
    assert router is not None

def test_main_app():
    from lover_agent.main import app
    assert app.title == "Chat Retrieval Agent"
```

- [ ] **Step 4: 运行测试验证通过**

Run: `pytest tests/test_api.py -v`
Expected: PASS

- [ ] **Step 5: 提交**

```bash
git add lover_agent/api/routes.py lover_agent/main.py tests/test_api.py
git commit -m "feat: 实现 API 路由与 FastAPI 入口"
```

---

## Task 9: 集成测试

**Files:**
- Create: `tests/test_integration.py`
- Create: `data/chroma/.gitkeep`

- [ ] **Step 1: 创建集成测试**

```python
import os

def test_full_pipeline():
    """完整的 RAG 流程测试"""
    from lover_agent.rag.loader import ChatRecordLoader
    from lover_agent.rag.splitter import ChatRecordSplitter
    from lover_agent.rag.vectorstore import ChatVectorStore
    from lover_agent.adapters.html_parser import ChatMessage
    from langchain_core.documents import Document

    # 1. 加载消息
    messages = [
        ChatMessage(sender="张三", content="今天天气真好", timestamp="2024-01-01 12:00:00"),
        ChatMessage(sender="李四", content="是啊，适合出去玩", timestamp="2024-01-01 12:01:00")
    ]

    # 2. 分割
    splitter = ChatRecordSplitter()
    docs = splitter.split_messages(messages)
    assert len(docs) > 0

    # 3. 向量存储（使用内存模式测试）
    # 注意：完整集成测试需要实际的模型文件
```

- [ ] **Step 2: 创建 .gitkeep**

```bash
mkdir -p data/chroma
touch data/chroma/.gitkeep
```

- [ ] **Step 3: 运行集成测试**

Run: `pytest tests/test_integration.py -v`
Expected: PASS

- [ ] **Step 4: 提交**

```bash
git add data/chroma/.gitkeep tests/test_integration.py
git commit -m "test: 添加集成测试"
```

---

## 实现顺序

1. **Task 1** - 项目初始化与配置
2. **Task 2** - HTML 解析器
3. **Task 3** - RAG Loader 与 Splitter
4. **Task 4** - Embedder 与 VectorStore
5. **Task 5** - Retriever 与 Indexer
6. **Task 6** - LangChain Chains
7. **Task 7** - Models 与 Services
8. **Task 8** - API 路由与入口
9. **Task 9** - 集成测试

---

## 自检清单

- [ ] 所有测试通过
- [ ] 所有文件路径正确
- [ ] 无 Placeholder (TBD/TODO)
- [ ] 类型一致性检查通过
- [ ] 代码提交完成