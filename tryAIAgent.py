from faiss_utils import index_document_to_faiss,load_faiss_from_local
from langchain.prompts import ChatPromptTemplate,PromptTemplate,SystemMessagePromptTemplate
from langchain_ollama import ChatOllama
from langchain_deepseek import ChatDeepSeek
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
#from operator import itemgetter
from dotenv import load_dotenv
#from openai import OpenAI
#import os
#os.environ["LANGCHAIN_TRACING_V2"] = "true"

load_dotenv()




# 将传入的文档转换成字符串的形式
 
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)



if __name__ == "__main__":
  

    #index_document_to_faiss(file_path ="C:/Users/Wells/Desktop/雨中草莓地.PDF",
                            #file_id=1)
    vector_db = load_faiss_from_local(1)
    print("加载成功")
    breakpoint()


    """llm = ChatDeepSeek(
                 model_name="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
                 api_base="https://api.siliconflow.cn/v1",
                 temperature=1)"""
    
    llm = ChatOllama(model="deepseek-r1:1.5b",temperature=0.5,format="json")
    

    system_prompt = """
    你是一个问答助手。请根据上下文详细的回答问题。
    如果你不知道答案，请说“我不太清楚”。
    

    <context>
    {context}
    </context>
    """

  
    rag_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "以下是我的问题：{input}"),
        ]
        )
    retriever = vector_db.as_retriever(search_kwargs={"k": 4})
    rag_chain = (
    {"context": retriever | format_docs, "input": RunnablePassthrough()}
    | rag_prompt
    | llm
    #| StrOutputParser()
    )   



    question ="概括一下剧情？"
    print(rag_chain.invoke(question).content)
   
    # Run
    #for result in rag_chain.stream("谁是杨晨？"):
        #print(result,end="",flush=True)
    #result=rag_chain.invoke("谈谈小张")
    #print(result)

    






    




   




    
    
    
