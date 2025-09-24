# src/llm_interface/openai_llm.py

import os
import json
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from typing import Dict, List
from dotenv import load_dotenv

load_dotenv() # 从.env 文件加载环境变量

class OpenAILLM:
    """
    封装与OpenAI API的所有交互，包括请求发送、错误处理和响应解析。
    """
    def __init__(self, model: str, temperature: float, max_tokens: int):
        self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def generate_captions(self, prompt: Dict[str, str]) -> List[str]:
        """
        调用OpenAI API生成描述，并包含强大的重试逻辑。

        Args:
            prompt (Dict[str, str]): 包含系统和用户提示词的字典。

        Returns:
            List[str]: 由LLM生成的描述列表。
        
        Raises:
            ValueError: 如果API返回的不是预期的JSON格式或内容为空。
        """
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": prompt['system']},
                    {"role": "user", "content": prompt['user']}
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                response_format={"type": "json_object"} # 强制JSON输出
            )
            
            response_text = response.choices.message.content
            data = json.loads(response_text)
            
            captions = data.get("captions")
            if not captions or not isinstance(captions, list):
                raise ValueError("LLM did not return a valid list of captions in the JSON object.")
                
            return captions

        except (json.JSONDecodeError, KeyError) as e:
            print(f"Error parsing LLM response: {e}")
            print(f"Raw response: {response_text}")
            raise ValueError("Failed to parse JSON from LLM response.")
        except Exception as e:
            print(f"An unexpected error occurred with OpenAI API: {e}")
            raise