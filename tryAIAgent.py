from langchain_community.document_loaders import PDFPlumberLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS

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
    embeddings = OllamaEmbeddings(model="nomic-embed-text")
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
    vector_db = FAISS.load_local("faiss_index", OllamaEmbeddings(model="nomic-embed-text"), allow_dangerous_deserialization=True)
    
    return vector_db

if __name__ == "__main__":
    print("Starting...\n")
    
    # 加载并分割文档
    chunks = doc_loader()
    print("切片完成\n")
    
    # 加载向量数据库
    vector_db = loading_vector_db()

    # 执行相似度搜索并打印结果
    docs = vector_db.similarity_search("W630")
    print(docs[0].page_content)