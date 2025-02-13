from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
import os
from langchain_community.document_loaders import PDFPlumberLoader

# 初始化文本分割器，用于将大型文本分割成较小的块
text_splitter = RecursiveCharacterTextSplitter(chunk_size=5000, chunk_overlap=500)
# 初始化嵌入模型，用于将文本转换为向量表示
embeddings = OllamaEmbeddings(model="nomic-embed-text:latest")

def load_and_split_document(file_path: str) -> List[Document]:
    """
    加载并分割文档。
    
    根据文件路径加载文档，并将其分割成多个较小的文档块。
    
    参数:
    - file_path: str - 文件路径
    
    返回:
    - List[Document] - 分割后的文档块列表
    """
    if file_path.endswith('.pdf') or file_path.endswith('.PDF') :
        loader = PDFPlumberLoader(file_path)
    else:
        raise ValueError("不受支持的格式")

    documents = loader.load()
    return text_splitter.split_documents(documents)

def index_document_to_faiss(file_path: str, file_id: int) -> bool:
    """
    将文档索引到FAISS数据库中。
    
    加载并分割文档，然后将分割后的文档块和它们的嵌入保存到本地的FAISS数据库中。
    
    参数:
    - file_path: str - 文件路径
    - file_id: int - 文件ID，用于标识文档
    
    返回:
    - bool - 表示操作是否成功的布尔值
    """
    try:
        # 加载并分割文档
        splits = load_and_split_document(file_path)
        
        # 将文档块转换为向量并保存到FAISS数据库
        vectorDB = FAISS.from_documents(splits, embeddings)
        vectorDB.save_local(os.path.join('faiss_index', str(file_id)))
        return True
    except Exception as e:
        print(f"创建\保存向量库失败: {e}")
        return False

def load_faiss_from_local(file_id:int):
    """
    从本地加载FAISS数据库。
    
    根据文件ID从本地加载FAISS数据库，并返回加载的数据库。
    
    参数:
    - file_id: int - 文件ID，用于标识文档
    
    返回:
    - FAISS - 加载的FAISS数据库
    _Note__: 如果加载失败，将返回False。  
    """
    try:
        # 从本地加载FAISS数据库
        vectorDB = FAISS.load_local(os.path.join('faiss_index', str(file_id)), embeddings,
                                    allow_dangerous_deserialization=True)
        return vectorDB
    except Exception as e:
        print(f"加载向量库失败: {e}")
        return False