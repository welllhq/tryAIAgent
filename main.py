from faiss_utils import index_document_to_faiss,load_faiss_from_local
from langchain_utils import recreate_question,get_rag_answer
from dotenv import load_dotenv

import os
#os.environ["LANGCHAIN_TRACING_V2"] = "true"

load_dotenv()



"""history = [
       ("user","谁是李白"),
       ("ai","李白是唐朝的诗人。"),
    ]"""

if __name__ == "__main__":
    
    #index_document_to_faiss(file_path ="C:/Users/Wells/Desktop/雨中草莓地.PDF",
                            #file_id=1)

    #print(get_rag_answer("谁是杨晨？",1))

    # Run
    result = get_rag_answer("谁是杨晨？",1,stream_flag=True)
    for chunk in result:
        print(chunk,end="",flush=True)
    breakpoint()
    



    




   




    
    
    
