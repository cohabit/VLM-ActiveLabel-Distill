import os

class YOLOExporter:
    """阶段4：高质量数据转存为 YOLO 格式，驱动本地小模型迭代自训练 (Distillation)"""

    def __init__(self, class_mapping: dict = None):
        # 类别名称映射为类别 ID，例如 {"汉堡": 0, "薯条": 1}
        self.class_mapping = class_mapping or {"默认目标": 0}

    def convert_to_yolo_line(self, box: list, label: str, img_width: int, img_height: int) -> str:
        """
        将绝对坐标 [xmin, ymin, xmax, ymax] 转换为 YOLO 格式:
        <class_id> <x_center> <y_center> <width> <height> (均归一化至 0~1)
        """
        if label not in self.class_mapping:
            self.class_mapping[label] = len(self.class_mapping)

        class_id = self.class_mapping[label]
        xmin, ymin, xmax, ymax = box

        # 计算中心点与宽高
        box_w = xmax - xmin
        box_h = ymax - ymin
        x_center = xmin + box_w / 2.0
        y_center = ymin + box_h / 2.0

        # 归一化至 0 ~ 1
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        w_norm = box_w / img_width
        h_norm = box_h / img_height

        return f"{class_id} {x_center_norm:.6f} {y_center_norm:.6f} {w_norm:.6f} {h_norm:.6f}"

    def save_yolo_annotation(self, output_txt_path: str, lines: list):
        """保存为 YOLO 专用的 .txt 标签文件"""
        os.makedirs(os.path.dirname(output_txt_path), exist_ok=True)
        with open(output_txt_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

    def generate_dataset_yaml(self, dataset_dir: str, output_yaml_path: str):
        """生成 YOLOv8 训练所需的 dataset.yaml 配置文件"""
        yaml_content = f"""# YOLOv8 自动标注蒸馏数据集配置
path: {dataset_dir}
train: images/train
val: images/val

names:
"""
        for name, cid in self.class_mapping.items():
            yaml_content += f"  {cid}: {name}\n"

        with open(output_yaml_path, "w", encoding="utf-8") as f:
            f.write(yaml_content)
        return output_yaml_path
