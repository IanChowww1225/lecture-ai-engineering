# config.py
import os
from dotenv import load_dotenv

# 加载环境变量
load_dotenv()

# 模型配置
MODEL_NAME = "google/gemma-2-2b-jpn-it"
HUGGINGFACE_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

# 数据库配置
DATABASE_FILE = "chat_history.db"

# 应用配置
MAX_HISTORY = 10
TEMPERATURE = 0.7
MAX_LENGTH = 1000