from langchain.prompts import ChatPromptTemplate,PromptTemplate
from langchain.prompts import SystemMessagePromptTemplate,MessagesPlaceholder
from langchain_community.vectorstores import FAISS
from langchain_core.runnables import RunnablePassthrough
from langchain_ollama import ChatOllama
from langchain_deepseek import ChatDeepSeek
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from typing import List, Any
from faiss_utils import index_document_to_faiss,load_faiss_from_local

#加载环境变量
load_dotenv()

#llm = ChatOllama(model="deepseek-r1:1.5b",temperature=0.5)
llm = ChatDeepSeek(
    model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    #model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    api_base="https://api.siliconflow.cn/v1",
    temperature=1)

#系统提示词，用于生成完整的对话格式
system_prompt = """
    你是一个问答助手。请根据上下文详细的回答问题。
    如果你不知道答案，请说“我不太清楚”。
    

    <context>
    {context}
    </context>
"""
# 创建一个聊天提示模板，用于生成系统提示和用户问题的对话格式
rag_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "以下是我的问题：{input}"),
    ]
    )

# 系统提示，用于生成一个独立的问题，不依赖于聊天历史
contextualize_q_system_prompt = (
    "给予聊天历史和最新的的用户问题， "
    "如果用户问题引用了聊天历史中的上文，则"
    "生成一个无需聊天历史记录也可理解的独立问题。"
    "不要回答问题，如有必要就重构问题后直接返回结果，否则原样返回。 "
    )

# 创建一个聊天提示模板，用于处理聊天历史和当前问题，以生成上下文完整的独立问题
contextualize_q_prompt = ChatPromptTemplate.from_messages([
    ("system", contextualize_q_system_prompt),
    MessagesPlaceholder("chat_history"),
    ("human", "{input}"),
])

# 用于格式化文档内容
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

# 根据聊天历史重新创建问题的函数
def recreate_question(question:str, chat_history:list = None, *, stream_flag:bool = False) -> Any:
    """
    重新生成问题函数。

    根据提供的问题和聊天历史记录，使用预定义的链（包括上下文提示、语言模型和字符串输出解析器）
    来重新生成或优化问题。

    参数:
    - question (str): 需要重新生成的问题。
    - chat_history (list, 可选): 聊天历史记录，用于提供上下文信息。默认为None。
    - stream_flag (bool, 可选): 是否以流式输出结果。默认为False。

    返回:
    - Any: 根据stream_flag的值，返回重新生成的问题的流式输出或调用结果。如果发生异常，返回False。
    """
    # 如果没有提供聊天历史记录，则初始化为空列表
    if chat_history is None:
        chat_history = []
    
    # 定义用于重新生成问题的链
    qa_recreate_chain = (
        contextualize_q_prompt  # 上下文提示，为问题生成提供背景信息
        | llm  # 语言模型，用于生成问题
        | StrOutputParser()  # 字符串输出解析器，解析生成的输出
    )
    
    try:
        # 根据stream_flag的值选择合适的执行方式
        if stream_flag:
            return qa_recreate_chain.stream({"chat_history": chat_history, "input": question})
        else:
            return qa_recreate_chain.invoke({"chat_history": chat_history, "input": question})
    except Exception as e:
        # 异常处理，打印错误信息并返回False
        print(e)
        return False

def get_rag_answer(input:str, file_id:int, *, stream_flag:bool = False) -> Any:
    """
    根据输入问题和指定的文件ID，使用RAG（Retrieval-Augmented Generation）模式生成答案。
    
    参数:
    - input (str): 输入的问题文本。
    - file_id (int): 指定的文件ID，用于从本地加载FAISS索引。
    - stream_flag (bool, 可选): 是否使用流式处理输出答案，默认为False。
    
    返回:
    - Any: 根据stream_flag的值，可能返回生成的答案文本或流式处理的输出。
    """
    # 从本地加载FAISS索引
    retrieve = load_faiss_from_local(file_id)
    # 创建一个检索器
    retriever = retrieve.as_retriever(search_kwargs={"k": 4})

    # 创建一个链式处理流程，用于生成答案
    rag_chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | rag_prompt
        | llm
        | StrOutputParser()
    )
    try:
        # 根据stream_flag决定是否使用流式处理
        if stream_flag:
            return rag_chain.stream(input)
        else:
            return rag_chain.invoke(input)
    except Exception as e:
        # 打印异常信息并返回False
        print(e)
        return False