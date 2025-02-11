from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate,PromptTemplate,SystemMessagePromptTemplate
from langchain_ollama import ChatOllama
from langchain_deepseek import ChatDeepSeek
from langchain.chains import RetrievalQA
from langchain_core.runnables import RunnablePassthrough
#from openai import OpenAI
#from langchain import hub
#import os
#os.environ["LANGCHAIN_TRACING_V2"] = "true"
#os.environ["LANGCHAIN_API_KEY"] = ""

def doc_loader(address):
    """
    加载PDF文档，并将其分割成合适的块大小返回。
    
    Returns:
        List: 文档块的列表。
    """
    # 初始化PDF加载器并加载PDF文档
    loader = PDFPlumberLoader(address)
    documents = loader.load()
    
    # 初始化文本分割器，并将加载的文档分割成指定大小的块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=8000, chunk_overlap=500
    )
    chunks = text_splitter.split_documents(documents)
    
    return chunks

def embedding_docs(chunks):
    """
    创建文档嵌入，并将其存储在向量数据库中。
    
    Args:
        chunks (List): 文档块的列表。
        
    Returns:
        FAISS: 存储了文档嵌入的向量数据库。
    """
    # 初始化文档嵌入，并使用指定的模型创建向量数据库
    embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")
    vector_db = FAISS.from_documents(chunks, embeddings)
    
    # 将向量数据库保存到本地
    vector_db.save_local("faiss_index") 
    
    return vector_db

def loading_vector_db():
    """
    从本地加载向量数据库。
    
    Returns:
        FAISS: 从本地加载的向量数据库。
    """
    # 从指定路径加载向量数据库
    vector_db = FAISS.load_local("faiss_index", OllamaEmbeddings(model="nomic-embed-text:latest"), allow_dangerous_deserialization=True)
    
    return vector_db

# 将传入的文档转换成字符串的形式
 
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def doc_exec(address):
    doc=doc_loader(address)
    return embedding_docs(doc)

if __name__ == "__main__":
  
    print("正在处理文档…")
    #vector_db = doc_exec("C:/Users/Wells/Desktop/雨中草莓地.PDF")
    # 加载向量数据库
    vector_db = loading_vector_db()

    llm = ChatDeepSeek(api_key="",
                 model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                 api_base="https://api.siliconflow.cn/v1",
                 temperature=0)
    
    #llm = ChatOllama(model="deepseek-r1:1.5b",temperature=0.5)
    


    RAG_TEMPLATE = """
    你是一个问答助手。请根据上下文简练但详细的回答问题。
    可以发表自己想法。
    

    <context>
    {context}
    </context>

    请根据问题作答:

    {question}"""

    sys_prompt = SystemMessagePromptTemplate.from_template(RAG_TEMPLATE)
    rag_prompt = ChatPromptTemplate.from_messages([sys_prompt])

    retriever = vector_db.as_retriever(search_kwargs={"k": 5})

    qa_chain = (
    {"context": retriever | format_docs, "question": RunnablePassthrough()}
    | rag_prompt
    #| llm
        )

    question = input("请输入问题：")
   
    # Run
    for result in qa_chain.stream(question):
        print(result.to_messages()[0].content,end="",flush=True)
    #result=qa_chain.invoke(question)
    #print(result)

    






    




   




    
    
    
