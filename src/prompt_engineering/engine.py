# src/prompt_engineering/engine.py

from typing import Dict, Any

class PromptEngine:
    """
    根据属性字典和模板，动态构建发送给LLM的最终提示词。
    """
    def __init__(self, prompts_config: Dict[str, str]):
        """
        使用 prompts.yaml 中的模板初始化引擎。

        Args:
            prompts_config (Dict[str, str]): 包含系统和用户提示词模板的字典。
        """
        self.system_prompt = prompts_config['system_prompt']
        self.user_prompt_template = prompts_config['user_prompt']

    def build_prompt(self, attributes: Dict[str, Any], num_captions: int) -> Dict[str, str]:
        """
        填充用户提示词模板。

        Args:
            attributes (Dict[str, Any]): 从AttributeMapper生成的属性字典。
            num_captions (int): 需要生成的描述数量。

        Returns:
            Dict[str, str]: 包含系统和用户提示词的字典，可直接用于API调用。
        """
        # 使用属性字典填充模板中的占位符
        user_prompt = self.user_prompt_template.format(
            num_captions=num_captions,
            **attributes
        )
        return {"system": self.system_prompt, "user": user_prompt}