from faiss_utils import index_document_to_faiss,load_faiss_from_local
from langchain_utils import recreate_question,get_rag_answer
from pydantic_models import QueryInput, QueryResponse
from db_utils import insert_application_logs, get_chat_history, get_rag_chain
from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, HTTPException
import uuid
import os
#os.environ["LANGCHAIN_TRACING_V2"] = "true"

load_dotenv()

app = FastAPI()

#@app.post("/chat",response_model=QueryResponse)
def chat(query_input: QueryInput):
    session_id = query_input.session_id or str(uuid.uuid4())
    #logging.info(f"Session ID: {session_id}, User Query: {query_input.question}, Model: {query_input.model.value}")

    chat_history = get_chat_history(session_id)
    if chat_history:
        re_question = recreate_question(query_input.question, chat_history)
        answer = get_rag_answer(chat_history, query_input.file_id)
    else:
        answer = get_rag_answer(query_input.question,query_input.file_id)

    insert_application_logs(session_id, query_input.question, answer,query_input.model.value)
    #logging.info(f"Session ID: {session_id}, AI Response: {answer}")
    return QueryResponse(answer=answer, session_id=session_id,model=query_input.model)

if __name__ == "__main__":
    
    #index_document_to_faiss(file_path ="C:/Users/Wells/Desktop/雨中草莓地.PDF",
                            #file_id=1)

    #print(get_rag_answer("概况剧情",1))

    # Run
    result = get_rag_answer("谈谈故事的主人公",1,stream_flag=True)
    for chunk in result:
        print(chunk,end="",flush=True)
  
    



    




   




    
    
    
