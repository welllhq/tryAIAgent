import sqlite3
from datetime import datetime

# 定义数据库名称
DB_name = 'rag_app.db'

def connect_to_db():
    """
    连接到SQLite数据库
    :return: SQLite数据库连接对象
    """
    conn = sqlite3.connect(DB_name)
    conn.row_factory = sqlite3.Row
    return conn

def create_application_logs():
    """
    创建存储应用日志的数据库表
    """
    conn = connect_to_db()
    conn.execute('''CREATE TABLE IF NOT EXISTS application_logs
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     session_id TEXT,
                     user_query TEXT,
                     llm_response TEXT,
                     model_name TEXT,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.close()

def create_document_store():
    """
    创建存储文档的数据库表
    """
    conn = connect_to_db()
    conn.execute('''CREATE TABLE IF NOT EXISTS document_store
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     filename TEXT,
                     upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.close()

def insert_application_logs(session_id, user_query, llm_response, model_name):
    """
    插入应用日志记录
    :param session_id: 会话ID
    :param user_query: 用户查询
    :param llm_response: LLM响应
    :param model_name: 模型名称
    """
    conn = connect_to_db()
    conn.execute('INSERT INTO application_logs (session_id, user_query, llm_response, model_name) VALUES (?, ?, ?, ?)',
                 (session_id, user_query, llm_response, model_name))
    conn.commit()
    conn.close()

def get_chat_history(session_id):
    """
    获取聊天历史记录
    :param session_id: 会话ID
    :return: 包含聊天消息的列表
    """
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute('SELECT user_query, llm_response FROM application_logs WHERE session_id = ? ORDER BY created_at', (session_id,))
    messages = []
    for row in cursor.fetchall():
        messages.extend([
            {"role": "human", "content": row['user_query']},
            {"role": "ai", "content": row['llm_response']}
        ])
    conn.close()
    return messages

def insert_document_record(filename):
    """
    插入文档记录
    :param filename: 文件名
    :return: 插入记录的ID
    """
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute('INSERT INTO document_store (filename) VALUES (?)', (filename,))
    file_id = cursor.lastrowid
    conn.commit()
    conn.close()
    return file_id

def delete_document_record(file_id):
    """
    删除文档记录
    :param file_id: 文件记录ID
    :return: 删除操作的结果
    """
    conn = connect_to_db()
    conn.execute('DELETE FROM document_store WHERE id = ?', (file_id,))
    conn.commit()
    conn.close()
    return True

def get_all_documents():
    """
    获取所有文档记录
    :return: 包含所有文档信息的列表
    """
    conn = connect_to_db()
    cursor = conn.cursor()
    cursor.execute('SELECT id, filename, upload_timestamp FROM document_store ORDER BY upload_timestamp DESC')
    documents = cursor.fetchall()
    conn.close()
    return [dict(doc) for doc in documents]

# 初始化数据库表
create_application_logs()
create_document_store()