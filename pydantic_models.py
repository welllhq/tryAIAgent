from pydantic import BaseModel, Field
from enum import Enum
from datetime import datetime


class ModelName(str, Enum):
    DEEPSEEK_7B = "deepseek_7B"
    DEEPSEEK_1_5B = "deepseek_1.5B"

class QueryInput(BaseModel):
    question: str
    session_id: str = Field(default=None)
    file_id: int = Field(default=1)
    model_name: ModelName = Field(default=ModelName.DEEPSEEK_7B)



class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model_name:ModelName
    
    