from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
import json
from langchain.vectorstores import FAISS
from langchain_community.embeddings import OllamaEmbeddings  


# 加载PDF文档
loader = PyMuPDFLoader("C:\\Users\\QL00123\\Desktop\\s7-200_SMART_system_manual_zh-CHS.pdf")
documents = loader.load()

# 分块文档
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=8000,
    chunk_overlap=200,
    separators=["\n\n", "\n", "。", " ", ""]
)
chunks = text_splitter.split_documents(documents)
#print(f"分块后的文档数量：{len(chunks)}")

def get_embedding(chunk):
    url = "https://api.siliconflow.cn/v1/embeddings"
    payload = {
    "model": "BAAI/bge-m3",
    "input": chunk,
    "encoding_format": "float"
    }
    headers = {
    "Authorization": "Bearer sk-wmtuhhunizaqtzlrdlbzmyzmlcuymeriuzcbjmaslkwovpgu",
    "Content-Type": "application/json"
    }
    response = requests.request("POST", url, json=payload, headers=headers).json()
    embedding = response['data'][0]['embedding']
    return embedding


#print(get_embedding(chunks[0].page_content))

embedding = HuggingFaceEmbeddings()
vector_db = FAISS.from_documents(chunks,embedding)
vector_db.save_local("faiss_index")  # 保存索引供后续使用
print("索引保存成功")
"""# 创建OpenAI的LLM
llm = OpenAI(
    openai_api_key="sk-wmtuhhunizaqtzlrdlbzmyzmlcuymeriuzcbjmaslkwovpgu",
    model="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
)

# 创建检索链
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_db.as_retriever(),
)

# 示例问题
question = "如何使用这个功能？"
response = qa_chain.run(question)
print(response)"""