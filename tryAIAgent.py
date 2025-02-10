from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.prompts import ChatPromptTemplate
#from langchain_ollama import OllamaLLM
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
#from langchain_core.runnables import RunnablePassthrough
#from langchain import hub
#import os
#os.environ["LANGCHAIN_TRACING_V2"] = "true"
#os.environ["LANGCHAIN_API_KEY"] = ""

def doc_loader():
    """
    加载PDF文档，并将其分割成合适的块大小返回。
    
    Returns:
        List: 文档块的列表。
    """
    # 初始化PDF加载器并加载PDF文档
    loader = PDFPlumberLoader("C:\\Users\\Wells\\Desktop\\USR630S.pdf")
    documents = loader.load()
    
    # 初始化文本分割器，并将加载的文档分割成指定大小的块
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=8000, chunk_overlap=200
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



if __name__ == "__main__":
    print("Starting...\n")
    
    # 加载并分割文档
    #chunks = doc_loader()
    #print("切片完成\n")
    
    # 加载向量数据库
    vector_db = loading_vector_db()

    llm = ChatOpenAI(api_key="",
                 model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                 base_url="https://api.siliconflow.cn/v1")
    
    prompt = """
    请严格根据以下上下文回答问题,力求答案简洁：
    {context}

    问题：{question}
    如果上下文不相关，请回答“我不确定，请联系客服”。
    答案：
    """
  
    rag_prompt = ChatPromptTemplate([("system",prompt)])
    retriever = vector_db.as_retriever()
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": rag_prompt},
        return_source_documents=True
        )
    question = "如何开启wlan功能？"
    result = qa_chain.invoke({"query": question})
    print(result["result"])



   




    
    
    
