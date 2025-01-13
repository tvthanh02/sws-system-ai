from typing import Optional
from pydantic import BaseModel
        # Định nghĩa schema cho dữ liệu yêu cầu (request)
class ProcessRequest(BaseModel):
    text: str  # Trường văn bản (string)

# Định nghĩa schema cho phản hồi (response)
class ProcessResponse(BaseModel):
    title: str
    isAntiState: bool
    generated_text: Optional[str]
