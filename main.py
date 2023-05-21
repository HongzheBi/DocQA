from qa import DocQA
from config.model_config import *
import os
os.chdir(os.getcwd() + '/data/')


if __name__ == "__main__":
    file_path = "financial_research_reports"
    vector_store_path = ""
    local_doc_qa = DocQA(llm_model=LLM_MODEL, embedding_model=EMBEDDING_MODEL, embedding_device=EMBEDDING_DEVICE)
    if LOAD_EMBEDDING == True:
        retriever = local_doc_qa.load_VectorDB(vector_store_path)
    else:
        retriever = local_doc_qa.init_knowledge_vector_store(file_path)
    # 循环输入查询，直到输入 "exit"
    while True:
        query = input("请输入问题（exit退出）：")

        if query == 'exit':
            print('exit')
            break

        docs = retriever.get_relevant_documents(query)
      
        prompt = local_doc_qa.generate_prompt(query, docs)
        response = local_doc_qa.llm._call(prompt=prompt)
        print("Query:" + query + '\nAnswer:' + response + '\n') 