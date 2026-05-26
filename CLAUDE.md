# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

基于 LangChain 的聊天记录检索 Agent，支持从 HTML 导出的聊天记录中进行 RAG（检索增强生成）。

## 常用命令

```bash
# 安装依赖
pip install -r requirements.txt

# 运行测试
pytest

# 运行单个测试文件
pytest tests/test_html_parser.py

# 启动开发服务器
python main.py
# 或使用 uvicorn
uvicorn main:app --reload
```

## 技术栈

- **Agent 框架**: LangChain >= 0.3.0
- **向量数据库**: ChromaDB >= 0.5.0
- **嵌入模型**: BGE (bge-small-zh，本地免费模型)
- **LLM**: 可配置（默认 GPT-4o-mini）
- **Web 框架**: FastAPI >= 0.115.0
- **配置管理**: .env + Pydantic Settings

## 项目结构

```
lover_agent/
├── adapters/
│   └── html_parser.py      # HTML 聊天记录解析（WeChatHTMLParser）
├── config/
│   └── settings.py         # 配置管理，读取 .env 环境变量
├── utils/
│   └── logger.py           # 日志工具（get_logger 函数）
├── tests/
│   └── test_html_parser.py # HTML 解析器测试
├── docs/superpowers/       # 设计文档和计划
│   ├── specs/              # 设计规范
│   └── plans/              # 实施计划
├── dataset/                # 聊天记录数据
├── main.py                 # FastAPI 入口（当前为 stub）
└── requirements.txt
```

## 核心模块

### adapters/html_parser.py

`WeChatHTMLParser` 类解析微信导出的 HTML 格式聊天记录：
- HTML 结构：`#chat-container` → `.item.item-left/right` → `.bubble.bubble-left/right`
- `ChatMessage` dataclass：包含 sender, content, timestamp, session_id 字段
- `parse(html_content)` → `List[ChatMessage]`
- `parse_file(file_path)` → `List[ChatMessage]`

### config/settings.py

`Settings` 类继承 `BaseSettings`，从 `.env` 读取配置：
- `chroma_db_path`: Chroma 数据库路径（默认 `./data/chroma`）
- `chroma_collection_name`: Collection 名称（默认 `chat_records`）
- `embedding_model`: 嵌入模型名称（默认 `bge-small-zh`）
- `llm_base_url`, `llm_api_key`, `llm_model`: LLM 配置

### utils/logger.py

`get_logger(name)` 函数返回配置好的 Logger 实例，支持 `LOG_LEVEL` 环境变量。

## 配置

环境变量配置文件：`.env.example`

```env
CHROMA_DB_PATH=./data/chroma
CHROMA_COLLECTION_NAME=chat_records
EMBEDDING_MODEL=bge-small-zh
EMBEDDING_MODEL_PATH=./models/bge-small-zh
LLM_BASE_URL=https://api.openai.com/v1
LLM_API_KEY=your-api-key-here
LLM_MODEL=gpt-4o-mini
LOG_LEVEL=INFO
```

## 设计文档

详细设计规范见 `docs/superpowers/specs/2026-05-26-chat-retrieval-agent-design.md`，包含：
- 完整的项目结构规划
- RAG 模块设计（loader, splitter, embedder, vectorstore, retriever, indexer）
- LangChain chains 设计
- API 接口设计

## 注意事项

- 项目尚在开发中，`main.py` 为入口 stub
- `adapters/` 目录用于外部服务适配层
- 测试使用 pytest 和 pytest-asyncio