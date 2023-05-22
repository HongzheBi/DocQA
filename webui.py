import gradio as gr
import logging
from qa import DocQA
from config.model_config import *

doc_qa = None

LOG_FORMAT = "%(levelname) -5s %(asctime)s" "-1d: %(message)s"
logger = logging.getLogger()
logger.setLevel(logging.INFO)
logging.basicConfig(format=LOG_FORMAT)

style = """
.gradio-header {
    background: #8821eb;
    color: #fff;
}
.gradio-inputs {
    background: #8821eb;
    color: #fff; 
}
.gradio-outputs {
    background: #8821eb;
    color: #fff;
} 
.gradio-footer {
    background: #8821eb;
    color: #fff;  
}
"""


def initialize():
    global doc_qa
    logger.info("开始初始化模型...")
    doc_qa = DocQA(llm_model=llm_model_dict[LLM_MODEL], embedding_model=embedding_model_dict[EMBEDDING_MODEL])
    logger.info("初始化完成!")

initialize()   

def chat(query, choice, filepath):
    if choice == "上传文件/文件夹":
        retriever = doc_qa.init_knowledge_vector_store(filepath) 
    else:
        retriever = doc_qa.load_VectorDB(filepath)
    docs = retriever.get_relevant_documents(query)
    prompt = doc_qa.generate_prompt(query, docs)
    response = doc_qa.llm._call(prompt=prompt)
    return response

# 初始化时不显示这两个组件
#upload = gr.components.File(type="file", label="上传文件/文件夹", visible=False) 
#vector_store_path = gr.components.File(type="file", label="选择知识库文件夹", visible=False) 
'''
# 定义显示upload组件的方法  
def show_upload():   
    upload = gr.components.File(type="file", label="上传文件/文件夹", visible=True) 
    vector_store_path = gr.components.File(type="file", label="选择知识库文件夹", visible=False)
    return upload, vector_store_path 

    
# 定义显示load_embedding组件的方法  
def show_load_embedding():
    upload = gr.components.File(type="file", label="上传文件/文件夹", visible=False) 
    vector_store_path = gr.components.File(type="file", label="选择知识库文件夹", visible=True)
    return  upload, vector_store_path

# 根据选择调用对应方法  
def visible(choice): 
    if choice == "上传文件/文件夹":
        return show_upload()
    else:
        return show_load_embedding()
'''
        
source_choicer = gr.components.Radio(choices=["上传文件/文件夹", "使用已有知识库"],  
                                     label="选择知识库来源", interactive=True)

filepath =  gr.components.File(type="file", label="上传文件/文件夹", visible=True) 

query = gr.components.Textbox(lines=1, label="输入查询")

outputs = gr.outputs.Textbox(label="回复")  

gr.Interface(fn=chat, inputs=[source_choicer, filepath, query],
      outputs=outputs, title="基于本地知识库检索和LLM轻量化微调的问答系统", style=style).launch(  
            server_name='0.0.0.0',                         
            server_port=7860,
            show_api=False,
            share=True,  
            inbrowser=False,
)