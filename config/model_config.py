import torch.cuda
import torch.backends
import os
import logging

LOG_FORMAT = "%(levelname) -5s %(asctime)s" "-1d: %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format=LOG_FORMAT)

embedding_model_dict = {
    "bert-base-chinese" : "bert-base-chinese",
    "ernie-base": "nghuyong/ernie-3.0-base-zh",
    "text2vec": "GanymedeNil/text2vec-large-chinese",
    "our_model" : "HongzheBi/MPNet-finetune"
}

# Embedding model name
EMBEDDING_MODEL = "text2vec"

# Embedding running device
EMBEDDING_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# supported LLM models
llm_model_dict = {
    "chatglm-6b-int4": "THUDM/chatglm-6b-int4",
    "chatglm-6b-int8": "THUDM/chatglm-6b-int8",
    "chatglm-6b": "THUDM/chatglm-6b",
}

# LLM model name
LLM_MODEL = "chatglm-6b"

# LLM lora path，默认为空，如果有请直接指定文件夹路径
LLM_LORA_PATH = "HongzheBi/ChatGLM-6b-Lora"

#LLM_LORA_PATH = ''
USE_LORA = True if LLM_LORA_PATH else False

# LLM streaming reponse
STREAMING = True

# Use p-tuning-v2 PrefixEncoder
USE_PTUNING_V2 = True
LLM_PTUNING_V2_PATH = os.getcwd() + '/finetune/ptuning/'

# LLM running device
LLM_DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

VS_ROOT_PATH = os.path.join(os.path.dirname(os.path.abspath('__file__')), "data/vector_store")
#VS_ROOT_PATH = "vector_store"

UPLOAD_ROOT_PATH = os.path.join(os.path.dirname(os.path.abspath('__file__')), "data")

#检索Embedding数量
VECTOR_SEARCH_TOP_K = 5

# 是否持久化存储
PERSIST_EMBEDDING = True

# 是否加载本地已有知识库
LOAD_EMBEDDING = False

# 已有知识库位置
PERSIST_DIRECTORY = None

# 基于上下文的prompt模版，请务必保留"{question}"和"{context}"
PROMPT_TEMPLATE1 = """请注意：请谨慎评估query与提示的Context信息的相关性，只根据本段输入文字信息的内容进行回答，如果query与提供的材料无关，请回答"我不知道"，另外也不要回答无关答案：
        Context: {context}
        Question: {question}
        Answer:
    """

PROMPT_TEMPLATE = """已知信息：
{context} 

根据上述已知信息，简洁和专业的来回答用户的问题。如果无法从中得到答案，请说 “根据已知信息无法回答该问题” 或 “没有提供足够的相关信息”，不允许在答案中添加编造成分，答案请使用中文。 
问题是：{question}
回答："""

logger.info(f"""
loading model config
llm device: {LLM_DEVICE}
embedding device: {EMBEDDING_DEVICE}
""")

'''
# 文本分句长度
SENTENCE_SIZE = 100
# 匹配后单段上下文长度
CHUNK_SIZE = 250
# LLM input history length
LLM_HISTORY_LEN = 3
# 知识检索内容相关度 Score, 数值范围约为0-1100，如果为0，则不生效，经测试设置为小于500时，匹配结果更精准
VECTOR_SEARCH_SCORE_THRESHOLD = 0
# 是否开启跨域，默认为False，如果需要开启，请设置为True
# is open cross domain
OPEN_CROSS_DOMAIN = False
'''