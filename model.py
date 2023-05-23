from transformers import AutoTokenizer, AutoModel, AutoConfig
from typing import List, Dict, Optional
from langchain.llms.base import LLM
from peft import PeftModel
from config.model_config import *
import torch
import json


def torch_gc():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    

class ChatGLM_6B_PEFT(LLM):
    max_token: int = 10000
    temperature: float = 0.8
    top_p = 0.9
    model: object = None
    tokenizer: object = None
    history_len = 10

    @property
    def _llm_type(self) -> str:
        return "CHatGLM-6B-PEFT"
    
    def _call(self, prompt:str, streaming:bool = True) -> str:
        for inum, (stream_response, _) in enumerate(self.model.stream_chat(
                    self.tokenizer,
                    prompt,
                    max_length=self.max_token,
                    temperature=self.temperature,
                    top_p=self.top_p,
            )):
            torch_gc()
            yield stream_response  
    
    def load_model(
            self,
            llm_device=LLM_DEVICE,
            model_name_or_path: str = "THUDM/chatglm-6b",
            use_ptuning_v2 = False,
            use_lora = False,
            device_map: Optional[Dict[str, int]] = None,
            **kwargs
    ):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)   
        model_config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

        if use_ptuning_v2:
            try:
                prefix_encoder_file = open(LLM_PTUNING_V2_PATH + 'config.json', 'r')
                prefix_encoder_config = json.loads(prefix_encoder_file.read())
                prefix_encoder_file.close()
                model_config.pre_seq_len = prefix_encoder_config['pre_seq_len']
                #model_config.prefix_projection = prefix_encoder_config['prefix_projection']
            except Exception as e:
                logger.error(f"加载PrefixEncoder config.json失败: {e}")

        self.model = AutoModel.from_pretrained(model_name_or_path, config=model_config, trust_remote_code=True, **kwargs).cuda()

        if LLM_LORA_PATH and use_lora:
          self.model = PeftModel.from_pretrained(self.model, LLM_LORA_PATH).half().to(llm_device)
          print("LoRA模型权重加载成功")

        if use_ptuning_v2:
            try:
                prefix_state_dict = torch.load(LLM_PTUNING_V2_PATH + 'adapter_model.bin')
                new_prefix_state_dict = {}
                for k, v in prefix_state_dict.items():
                    if k.startswith("transformer.prefix_encoder."):
                        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
                self.model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)
                self.model.transformer.prefix_encoder.float()
                print("PTuning-v2模型权重加载成功")
            except Exception as e:
                logger.error(f"加载PrefixEncoder模型参数失败:{e}")
        
        self.model = self.model.eval()