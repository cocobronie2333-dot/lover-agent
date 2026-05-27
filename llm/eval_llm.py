import json
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import JsonOutputParser
from langchain_openai import ChatOpenAI
from llm.llm import LLM  

# 初始化高性价比的裁判模型（强烈推荐 GPT-4o-mini 或 Gemini-1.5-Flash）
eval_llm = LLM().llm

# 设计裁判专用的评分 Prompt
eval_prompt_template = ChatPromptTemplate.from_messages([
    ("system", (
        "你是一位严谨的 RAG (检索增强生成) 系统评测专家。\n"
        "请根据用户提问（Query）和召回的上下文（Context），评估该文本片段是否成功“召回”了用户需要的信息，特别是多媒体和静态资源（图片、视频、链接、表情包）。\n\n"
        "【评分标准 (Rubric)】：\n"
        "5分 (完美召回)：上下文完美包含用户需要的内容，或者精准包含了用户索要的多媒体资源路径/特征。\n"
        "4分 (强相关)：上下文高度相关，能用于回答问题，但多媒体资源特征稍微不够直观。\n"
        "3分 (弱相关)：上下文与问题有一定语义关联（如提到了相同的话题），但无法直接回答用户具体要找的静态资源文件。\n"
        "1分 (完全不相关)：上下文与用户提问完全风马牛不相及。\n\n"
        "【输出格式限制】：\n"
        "必须输出严格的 JSON 格式，包含以下字段，严禁包含任何 Markdown 格式包裹（如 ```json ）：\n"
        "{{\n"
        "  \"score\": 整数分值(1, 3, 4, 5),\n"
        "  \"reason\": \"一句话陈述打分理由（必须指出多媒体资源是否成功匹配）\"\n"
        "}}"
    )),
    ("user", "【用户提问】: {query}\n\n【召回上下文】:\n{context}")
])

# 转换为运行链
eval_chain = eval_prompt_template | eval_llm | JsonOutputParser()