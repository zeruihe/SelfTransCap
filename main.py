# main.py

import pandas as pd
import yaml
from tqdm import tqdm
from src.data_processing.mapper import AttributeMapper
from src.prompt_engineering.engine import PromptEngine
from src.llm_interface.openai_llm import OpenAILLM

def load_config(config_path: str):
    """
    加载YAML配置文件。
    
    Args:
        config_path (str): 配置文件路径
        
    Returns:
        dict: 配置文件内容
    """
    with open(config_path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)

def main():
    """
    主函数，编排整个TransCap数据生成流程。
    """
    print("Loading configurations...")
    config = load_config('configs/config.yaml')
    mappings = load_config('configs/mappings.yaml')
    prompts = load_config('configs/prompts.yaml')
    
    print("Step 1: Mapping raw data to structured attributes...")
    mapper = AttributeMapper(config, mappings)
    attributes_df = mapper.process_data()
    
    print("Step 2: Initializing LLM interface and prompt engine...")
    llm = OpenAILLM(
        model=config['llm']['model'],
        temperature=config['llm']['temperature'],
        max_tokens=config['llm']['max_tokens']
    )
    prompt_engine = PromptEngine(prompts)
    
    num_captions = config['generation']['num_captions_per_image']
    results = []
    
    print(f"Step 3: Generating {num_captions} captions for each of the {len(attributes_df)} images...")
    for _, row in tqdm(attributes_df.iterrows(), total=len(attributes_df), desc="Generating Captions"):
        attributes = row.to_dict()
        prompt = prompt_engine.build_prompt(attributes, num_captions)
        
        try:
            generated_captions = llm.generate_captions(prompt)
            
            # 确保生成的描述数量与要求一致
            if len(generated_captions) == num_captions:
                record = {"patientId": attributes['patientId']}
                for i, caption in enumerate(generated_captions):
                    record[f'describe_{i+1}'] = caption
                results.append(record)
            else:
                print(f"Warning: Expected {num_captions} captions for {attributes['patientId']}, but got {len(generated_captions)}. Skipping.")

        except Exception as e:
            print(f"Error processing {attributes['patientId']}: {e}. Skipping.")

    print("Step 4: Saving results to CSV...")
    output_df = pd.DataFrame(results)
    output_path = config['paths']['output_captions_csv']
    output_df.to_csv(output_path, index=False)
    
    print(f"Process complete. Captions saved to {output_path}")

if __name__ == "__main__":
    main()