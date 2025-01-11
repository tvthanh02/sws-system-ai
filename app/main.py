import os
import sys
# Thêm thư mục cha vào PYTHONPATH
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import uvicorn
from app.api.v1.text_generation_controller import setup_app

if __name__ == "__main__":
    app = setup_app()
    uvicorn.run(app, host="127.0.0.1", port=5000)