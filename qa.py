from langchain.embeddings import SentenceTransformerEmbeddings 
from langchain.vectorstores import Chroma
from langchain.docstore.document import Document
from datasets import load_dataset
import PyPDF2
import re
import os
from tqdm import tqdm
from model import ChatGLM_6B_PEFT, torch_gc
from config.model_config import *


# 加载文件
def load_file(file_name):
   if file_name[-3:] == "pdf":
      pdf_file = open(file_name, 'rb')
      pdf_reader = PyPDF2.PdfReader(pdf_file)
      text = ''
      for num in range(len(pdf_reader.pages)):
        page = pdf_reader.pages[num]
        text += page.extract_text()
   elif file_name[-3:] == "txt":
      with open(file_name, "r", encoding='utf-8') as f:
        text = str(f.read())             
   return text


# 自定义句子分段的方式，保证句子不被截断
def split_paragraph(text, persist_directory, max_length=300):
   text = text.replace('\n', '') 
   text = text.replace('\n\n', '') 
   text = re.sub(r'\s+', ' ', text)
   """
   将文章分段
   """
   # 首先按照句子分割文章
   sentences = re.split('(；|。|！|\!|\.|？|\?)',text) 
        
   new_sents = []
   for i in range(int(len(sentences)/2)):
      sent = sentences[2*i] + sentences[2*i+1]
      new_sents.append(sent)
   if len(sentences) % 2 == 1:
      new_sents.append(sentences[len(sentences)-1])
   # 按照要求分段
   paragraphs = []
   current_length = 0
   current_paragraph = ""
   for sentence in new_sents:
      sentence_length = len(sentence)
      if current_length + sentence_length <= max_length:
          current_paragraph += sentence
          current_length += sentence_length
      else:
          paragraphs.append(current_paragraph.strip())
          current_paragraph = sentence
          current_length = sentence_length
   paragraphs.append(current_paragraph.strip())
   documents = []
   metadata = {"source": persist_directory}
   for paragraph in paragraphs:
      new_doc = Document(page_content=paragraph, metadata=metadata)
      documents.append(new_doc)
   return documents


class DocQA:
    def __init__(self, 
           embedding_model = EMBEDDING_MODEL,
           embedding_device = EMBEDDING_DEVICE,
           llm_model = LLM_MODEL,
           llm_device = LLM_DEVICE,
           use_ptuning_v2 = USE_PTUNING_V2,
           use_lora = USE_LORA,
           file_name = None
        ):
        # 加载模型
        self.llm = ChatGLM_6B_PEFT()
        self.llm.load_model(model_name_or_path=llm_model_dict[LLM_MODEL], use_lora=use_lora, use_ptuning_v2=USE_PTUNING_V2)
        self.SBert = SentenceTransformerEmbeddings(model_name=embedding_model_dict[embedding_model], model_kwargs={'device':embedding_device})
        self.topk = VECTOR_SEARCH_TOP_K

    # 对文件embdding并持久化
    def init_knowledge_vector_store(self, filepath, vector_store_path=None):
        loaded_files = []
        failed_files = []
        if not vector_store_path:
            vector_store_path = os.path.join(VS_ROOT_PATH, filepath)
        if filepath[:9] == "HongzheBi":
            os.mkdir(filepath)
            dataset = load_dataset("text", data_files=filepath, cache_dir="./dataset")
            dataset.save_to_disk(os.getcwd() + "/data" + filepath[9:])
        elif isinstance(filepath, str):
          if not os.path.exists(filepath):
            print("路径不存在")
            return None
          elif os.path.isfile(filepath):
              file = os.path.split(filepath)[-1]
              try:
                doc = load_file(filepath)
                docs = split_paragraph(doc, vector_store_path)
                logger.info(f"{file} 已成功加载")
                loaded_files.append(filepath)
              except Exception as e:
                logger.error(e)
                logger.info(f"{file} 未能成功加载")
                return None
          elif os.path.isdir(filepath):
              docs = []
              for file in tqdm(os.listdir(filepath), desc="加载文件"):
                  fullfilepath = os.path.join(filepath, file)
                  try:
                      doc = load_file(fullfilepath)
                      docs += split_paragraph(doc, vector_store_path)
                      loaded_files.append(fullfilepath)
                  except Exception as e:
                      logger.error(e)
                      failed_files.append(file)
              if len(failed_files) > 0:
                  logger.info("以下文件未能成功加载：")
                  for file in failed_files:
                      logger.info(f"{file}\n")
        else:
            docs = []
            for file in filepath:
                try:
                    doc = load_file(file)
                    docs += split_paragraph(doc, vector_store_path)
                    logger.info(f"{file} 已成功加载")
                    loaded_files.append(file)
                except Exception as e:
                    logger.error(e)
                    logger.info(f"{file} 未能成功加载")
        if len(docs) > 0:
            logger.info("文件加载完毕，正在生成向量库")
            vectordb = Chroma.from_documents(documents=docs, embedding=self.SBert, persist_directory=vector_store_path)
            torch_gc()
            if PERSIST_EMBEDDING == True:
              vectordb.persist()
            vectordb = None
            return vector_store_path, loaded_files
        else:
            logger.info("文件均未成功加载，请检查依赖包或替换为其他文件再次上传。")
            return None, loaded_files

    # 加载向量数据库
    def load_VectorDB(self, vector_store_path):
      vectordb = Chroma(persist_directory=vector_store_path, embedding_function=self.SBert)
      retriever = vectordb.as_retriever(search_kwargs={"k": self.topk})
      return retriever       
    
    #根据Prompt模板生成Prompt
    def generate_prompt(self, query, docs):
        PromptTemplate = PROMPT_TEMPLATE
        context = "\n".join([doc.page_content for doc in docs])
        prompt = PromptTemplate.replace("{question}", query).replace("{context}", context)
        return prompt