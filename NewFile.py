import PyPDF2

from langchain.docstore.document import Document
from langchain.text_splitter import CharacterTextSplitter

def load_pdf(file_path):
    # 打开PDF文件
    with open(file_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        
        # 提取所有页面的文本
        text = ''
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text += page.extract_text()
    
    return text

def split_text(text):
    # 使用CharacterTextSplitter进行文本分块
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
    )
    chunks = text_splitter.split_text(text)
    return chunks

def main():
    # 加载PDF文件
    file_path = 'example.pdf'
    text = load_pdf(file_path)
    
    # 分块处理文本
    chunks = split_text(text)
    
    # 将每个分块转换为Document对象（可选）
    documents = [Document(page_content=chunk) for chunk in chunks]
    
    # 输出结果
    for i, doc in enumerate(documents):
        print(f"Chunk {i+1}:")
        print(doc.page_content)
        print("-" * 80)

if __name__ == "__main__":
    main()