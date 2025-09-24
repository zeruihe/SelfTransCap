# src/data_processing/mapper.py

import pandas as pd
from typing import Dict, Any, List

class AttributeMapper:
    """
    负责将原始的RSNA CSV数据转换为结构化的属性字典，
    为后续的提示词工程做准备。
    """
    def __init__(self, config: Dict[str, Any], mappings: Dict[str, Any]):
        """
        初始化映射器。

        Args:
            config (Dict[str, Any]): 主配置文件内容。
            mappings (Dict[str, Any]): 映射规则文件内容。
        """
        self.config = config
        self.mappings = mappings
        self.class_info_df = pd.read_csv(config['paths']['class_info_csv'])
        self.labels_df = pd.read_csv(config['paths']['labels_csv'])

    def _get_location_string(self, bboxes: List[Dict[str, float]]) -> str:
        """
        根据边界框列表，将其坐标转换为描述性位置字符串。
        
        该方法是TransCap方法中将结构化位置信息转化为自然语言描述的关键步骤。
        它读取`mappings.yaml`中定义的网格布局，动态地将数值坐标映射到
        "左上"、"中央"等语义化区域。
        """
        if not bboxes:
            return "N/A"

        grid_config = self.mappings['location_grid']
        rows, cols = grid_config['rows'], grid_config['cols']
        img_size = grid_config['image_size']
        names = grid_config['names']
        
        locations = set()
        for bbox in bboxes:
            center_x = bbox['x'] + bbox['width'] / 2
            center_y = bbox['y'] + bbox['height'] / 2
            
            row_idx = min(int(center_y / (img_size / rows)), rows - 1)
            col_idx = min(int(center_x / (img_size / cols)), cols - 1)
            
            locations.add(names[row_idx][col_idx])
            
        return ", ".join(sorted(list(locations)))

    def process_data(self) -> pd.DataFrame:
        """
        主处理函数，整合所有数据并生成属性。
        
        Returns:
            pd.DataFrame: 包含每个patientId及其对应属性字典的DataFrame。
        """
        # 合并两个信息表
        merged_df = pd.merge(self.class_info_df, self.labels_df, on='patientId', how='left')
        
        # 按patientId聚合边界框信息
        bbox_agg = merged_df.dropna(subset=['x']).groupby('patientId').apply(
            lambda group: group[['x', 'y', 'width', 'height']].to_dict('records'),
            include_groups=False
        ).reset_index(name='bboxes')
        
        # 获取每个patientId的唯一类别信息
        unique_class_df = self.class_info_df.drop_duplicates(subset=['patientId'])
        
        # 将聚合后的边界框信息与类别信息合并
        final_df = pd.merge(unique_class_df, bbox_agg, on='patientId', how='left')
        final_df['bboxes'] = final_df['bboxes'].apply(lambda d: d if isinstance(d, list) else [])

        # 生成属性字典
        attributes_list = []
        for _, row in final_df.iterrows():
            bboxes = row['bboxes']
            num_opacities = len(bboxes)
            
            attributes = {
                "patientId": row['patientId'],
                "modality": "Chest X-ray (CXR)",
                "pathology": self.mappings['class_to_pathology'].get(row['class'], "Unknown"),
                "num_opacities": num_opacities,
                "location_str": self._get_location_string(bboxes),
                "notes": "Multiple opacities are present." if num_opacities > 1 else "A single opacity is noted." if num_opacities == 1 else "No opacities detected."
            }
            attributes_list.append(attributes)
            
        return pd.DataFrame(attributes_list)