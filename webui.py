import os
import shutil
import gradio as gr
from qa import DocQA
from config.model_config import *


def get_vs_list():
    lst_default = ["新建知识库"]
    if not os.path.exists(VS_ROOT_PATH):
        return lst_default
    lst = os.listdir(VS_ROOT_PATH)
    if not lst:
        return lst_default
    lst.sort()
    return lst_default + lst


vs_list = get_vs_list()
doc_qa = DocQA()


def get_answer(query, vs_path, history, mode):
    if mode == "知识库问答" and os.path.exists(vs_path):
        retriever = doc_qa.load_VectorDB(vs_path)
        docs = retriever.get_relevant_documents(query)
        prompt = doc_qa.generate_prompt(query, docs)
        for response in doc_qa.llm._call(prompt=prompt):
            source = "\n\n"
            source += "".join([
                        f"""<details> <summary>出处 [{i + 1}] <a href="{doc.metadata["source"]}" target="_blank">{doc.metadata["source"]}</a> </summary>\n"""
                        f"""{doc.page_content}\n"""
                        f"""</details>\n"""
                        for i, doc in enumerate(docs)])
            yield history + [[query, response+source]], ""
    else:
        for response in doc_qa.llm._call(prompt=prompt):
            yield history + [[query, response]], ""


def get_vector_store(vs_id, files, sentence_size, history):
    vs_path = os.path.join(VS_ROOT_PATH, vs_id)
    filelist = []
    if not os.path.exists(os.path.join(UPLOAD_ROOT_PATH, vs_id)):
        os.makedirs(os.path.join(UPLOAD_ROOT_PATH, vs_id))
    if doc_qa.llm and doc_qa.SBert:
        if isinstance(files, list):
            for file in files:
                filename = os.path.split(file.name)[-1]
                shutil.move(file.name, os.path.join(UPLOAD_ROOT_PATH, vs_id, filename))
                filelist.append(os.path.join(UPLOAD_ROOT_PATH, vs_id, filename))
            logger.info(filelist)
            logger.info(vs_path)
            vs_path, loaded_files = doc_qa.init_knowledge_vector_store(filelist, vs_path)
        if len(loaded_files):
            file_status = f"已添加 {'、'.join([os.path.split(i)[-1] for i in loaded_files])} 内容至知识库，并已加载知识库，请开始提问"
        else:
            file_status = "文件未成功加载，请重新上传文件"
    else:
        file_status = "模型未完成加载，请先在加载模型后再导入文件"
        vs_path = None
    logger.info(file_status)
    return vs_path, None, history + [[None, file_status]]


def change_vs_name_input(vs_id, history):
    if vs_id == "新建知识库":
        return gr.update(visible=True), gr.update(visible=True), gr.update(visible=False), None, history
    else:
        file_status = f"已加载知识库{vs_id}"
        return gr.update(visible=False), gr.update(visible=False), gr.update(visible=True), os.path.join(VS_ROOT_PATH,vs_id), history + [[None, file_status]]


def change_mode(mode, history):
    if mode == "知识库问答":
        return gr.update(visible=True), gr.update(visible=False), history
    else:
        return gr.update(visible=False), gr.update(visible=False), history


def add_vs_name(vs_name, vs_list, chatbot):
    if vs_name in vs_list:
        vs_status = "与已有知识库名称冲突，请重新选择其他名称后提交"
        chatbot = chatbot + [[None, vs_status]]
        return gr.update(visible=True), vs_list, gr.update(visible=True), gr.update(visible=True), gr.update(
            visible=False), chatbot
    else:
        vs_status = f"""已新增知识库"{vs_name}",将在上传文件并载入成功后进行存储。请在开始对话前，先完成文件上传。 """
        chatbot = chatbot + [[None, vs_status]]
        return gr.update(visible=True, choices=[vs_name] + vs_list, value=vs_name), [vs_name] + vs_list, gr.update(
            visible=False), gr.update(visible=False), gr.update(visible=True), chatbot


block_css = """.importantButton {
    background: linear-gradient(45deg, #7e0570,#5d1c99, #6e00ff) !important;
    border: none !important;
}
.importantButton:hover {
    background: linear-gradient(45deg, #ff00e0,#8500ff, #6e00ff) !important;
    border: none !important;
}"""
webui_title = """
# 基于本地知识库检索和 LLM 轻量化微调的问答系统
"""
default_vs = vs_list[0] if len(vs_list) > 1 else "为空"
init_message = f"""欢迎使用 DocQA Web UI！
请在右侧切换模式，目前支持基于本地知识库问答或直接与 LLM 模型对话。
知识库问答模式，选择知识库名称后，即可开始问答，当前知识库为空，如有需要可以在选择知识库名称后上传文件/文件夹至知识库。
"""


with gr.Blocks(css=block_css) as demo:
    vs_path, vs_list = gr.State(
        os.path.join(VS_ROOT_PATH, vs_list[0]) if len(vs_list) > 1 else ""), gr.State(vs_list)
    gr.Markdown(webui_title)
    with gr.Tab("对话"):
        with gr.Row():
            with gr.Column(scale=10):
                chatbot = gr.Chatbot([[None, init_message]],  elem_id="chat-box", show_label=False).style(height=750)
                query = gr.Textbox(show_label=False,  placeholder="请输入提问内容，按回车进行提交").style(container=False)
            with gr.Column(scale=5):
                mode = gr.Radio(["LLM 对话", "知识库问答"], label="请选择使用模式", value="知识库问答", )
                knowledge_set = gr.Accordion("知识库设定", visible=False)
                vs_setting = gr.Accordion("配置知识库")
                mode.change(fn=change_mode, inputs=[mode, chatbot], outputs=[vs_setting, knowledge_set, chatbot])
                with vs_setting:
                    select_vs = gr.Dropdown(vs_list.value,  label="请选择要加载的知识库", interactive=True, value=vs_list.value[0] if len(vs_list.value) > 0 else None)
                    vs_name = gr.Textbox(label="请输入新建知识库名称，当前知识库命名暂不支持中文",  lines=1,  interactive=True, visible=True)
                    vs_add = gr.Button(value="添加至知识库选项", visible=True)
                    file2vs = gr.Column(visible=False)
                    with file2vs:
                        # load_vs = gr.Button("加载知识库")
                        gr.Markdown("向知识库中添加文件")
                        sentence_size = gr.Number(value=100, precision=0, label="文本入库分句长度限制", interactive=True, visible=True)
                        with gr.Tab("上传文件"):
                            files = gr.File(label="添加文件", file_types=['.txt', '.md', '.docx', '.pdf'],  file_count="multiple",  show_label=False)
                            load_file_button = gr.Button("上传文件并加载知识库")
                        with gr.Tab("上传文件夹"):
                            folder_files = gr.File(label="添加文件", file_count="directory",  show_label=False)
                            load_folder_button = gr.Button("上传文件夹并加载知识库")
                    vs_add.click(fn=add_vs_name,  inputs=[vs_name, vs_list, chatbot], outputs=[select_vs, vs_list, vs_name, vs_add, file2vs, chatbot])
                    select_vs.change(fn=change_vs_name_input, inputs=[select_vs, chatbot],  outputs=[vs_name, vs_add, file2vs, vs_path, chatbot])
                    load_file_button.click(get_vector_store,  show_progress=True, inputs=[select_vs, files, sentence_size, chatbot], outputs=[vs_path, files, chatbot], )
                    load_folder_button.click(get_vector_store,  show_progress=True, inputs=[select_vs, folder_files, sentence_size, chatbot],  outputs=[vs_path, folder_files, chatbot])
                    query.submit(get_answer,  [query, vs_path, chatbot, mode],  [chatbot, query])
(demo .queue(concurrency_count=3) .launch(server_name='0.0.0.0',  server_port=7860, show_api=False, share=True, inbrowser=False, debug=True))