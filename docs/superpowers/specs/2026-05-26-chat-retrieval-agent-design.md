# Chat Retrieval Agent 项目设计规范

## 项目概述

基于 LangChain 构建的聊天记录检索 Agent，支持从 HTML 导出的聊天记录中进行 RAG（检索增强生成），为 Web 应用提供智能对话检索能力。

## 技术栈

| 组件 | 技术选型 |
|------|----------|
| 编程语言 | Python |
| Agent 框架 | LangChain |
| 向量数据库 | Chroma |
| 嵌入模型 | BGE (免费本地模型) |
| LLM | 可配置（如 OpenAI GPT 系列） |
| Web 框架 | FastAPI |
| 配置管理 | .env + Pydantic Settings |

## 项目结构

```
lover_agent/
├── config/
│   └── settings.py           # 配置管理，读取 .env 环境变量
├── adapters/
│   ├── html_parser.py        # 解析微信/QQ 等导出的 HTML 聊天记录
│   └── chroma_client.py      # Chroma 客户端封装
├── rag/                      # RAG 核心模块
│   ├── loader.py             # HTML 文档加载
│   ├── splitter.py           # 文本分割策略
│   ├── embedder.py           # 嵌入模型（本地免费模型）
│   ├── vectorstore.py        # Chroma 向量存储管理
│   ├── retriever.py          # 检索器
│   └── indexer.py            # 构建索引
├── chains/
│   ├── retrieval_chain.py     # 检索链（LangChain）
│   └── conversational_chain.py  # 对话链
├── api/
│   └── routes.py             # FastAPI 路由
├── services/
│   └── chat_service.py       # 聊天服务
├── models/
│   └── schemas.py             # Pydantic 数据模型
├── utils/
│   └── logger.py             # 日志工具
├── main.py                   # FastAPI 入口
├── requirements.txt
├── .env                      # 环境变量（不提交到 git）
└── .env.example              # 环境变量模板
```

## 核心模块说明

### rag/

RAG 核心模块，负责文档处理和检索流程。

| 文件 | 功能 |
|------|------|
| `loader.py` | 加载 HTML 文件，解析聊天记录结构 |
| `splitter.py` | 按会话/时间/语义分割文本 |
| `embedder.py` | 使用本地嵌入模型进行向量化 |
| `vectorstore.py` | Chroma 向量存储的 CRUD 操作 |
| `retriever.py` | 检索器实现，支持混合检索 |
| `indexer.py` | 批量构建索引 |

### chains/

LangChain 链定义。

| 文件 | 功能 |
|------|------|
| `retrieval_chain.py` | RAG 检索链：查询 → 检索 → 组装上下文 |
| `conversational_chain.py` | 对话链：结合历史上下文生成回复 |

### adapters/

外部服务适配层。

| 文件 | 功能 |
|------|------|
| `html_parser.py` | HTML 聊天记录解析，提取用户、消息、时间 |
| `chroma_client.py` | Chroma 客户端封装，提供统一接口 |

## 数据流

### 索引构建流程

```
HTML 聊天记录文件
        ↓
    loader.py → 解析提取消息（用户、时间、文本）
        ↓
    splitter.py → 按会话/时间戳分割成 chunks
        ↓
    embedder.py → 使用 BGE 模型向量化
        ↓
    vectorstore.py → 存储到 Chroma
```

### 检索流程

```
用户查询文本
        ↓
    embedder.py → 向量化查询
        ↓
    vectorstore.py → 检索相似文档
        ↓
    retriever.py → 过滤、重排序
        ↓
    retrieval_chain.py → 组装提示词
        ↓
    LLM → 生成最终回复
```

## 配置管理

### .env 示例

```env
# 向量数据库
CHROMA_DB_PATH=./data/chroma
CHROMA_COLLECTION_NAME=chat_records

# 嵌入模型（本地免费模型）
EMBEDDING_MODEL=bge-small-zh
EMBEDDING_MODEL_PATH=./models/bge-small-zh

# LLM 配置
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your-api-key
LLM_MODEL=gpt-4o-mini

# 日志
LOG_LEVEL=INFO
```

### settings.py 结构

```python
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
```

## API 接口设计

### 基础接口

| 接口 | 方法 | 功能 |
|------|------|------|
| `/chat` | POST | 聊天对话（带 RAG 检索） |
| `/chat/history/{session_id}` | GET | 获取对话历史 |
| `/index` | POST | 构建索引（批量导入聊天记录） |
| `/index/{file_id}` | DELETE | 删除指定索引 |
| `/health` | GET | 健康检查 |

### 请求/响应模型

详见 `models/schemas.py`

## 约束与注意事项

1. **HTML 格式兼容**：需适配微信、QQ 等主流聊天工具导出的 HTML 格式
2. **离线优先**：嵌入模型使用本地免费模型（BGE），可联网获取更新
3. **敏感信息**：所有 API Key 等敏感配置必须通过 `.env` 管理，不硬编码
4. **向量数据库持久化**：Chroma 数据存储在本地，支持跨会话复用