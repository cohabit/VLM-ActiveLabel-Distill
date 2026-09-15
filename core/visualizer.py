import os
from PIL import Image, ImageDraw, ImageFont

class BoundingBoxVisualizer:
    """可视化引擎：在图像上精准绘制检测框与决策标签"""

    COLOR_MAP = {
        "AUTO_ADOPT": (34, 197, 94),     # 绿色：自动采纳 (高置信度)
        "HUMAN_REVIEW": (234, 179, 8),   # 黄色：人工审核 (边缘分歧)
        "DISCARD": (239, 68, 68)         # 红色：废弃
    }

    @classmethod
    def draw_annotations(cls, image_path: str, annotated_items: list, output_path: str):
        """
        绘制边界框与标签文本并保存到 output_path
        annotated_items 格式示例:
        [
            {
                "box": [xmin, ymin, xmax, ymax],
                "label": "汉堡",
                "action": "AUTO_ADOPT",
                "level": "Level-1"
            }
        ]
        """
        if not os.path.exists(image_path):
            return None

        img = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size

        # 尝试加载默认字体
        try:
            # 常见中文字体或默认
            font = ImageFont.truetype("simhei.ttf", 16)
        except Exception:
            font = ImageFont.load_default()

        for item in annotated_items:
            box = item.get("box", [])
            if len(box) != 4:
                continue

            xmin, ymin, xmax, ymax = box
            # 边界限制防止画出图外
            xmin = max(0, min(w - 1, xmin))
            ymin = max(0, min(h - 1, ymin))
            xmax = max(0, min(w - 1, xmax))
            ymax = max(0, min(h - 1, ymax))

            action = item.get("action", "AUTO_ADOPT")
            color = cls.COLOR_MAP.get(action, (34, 197, 94))
            label = item.get("label", "Object")
            level = item.get("level", "")

            # 1. 绘制加粗矩形框 (线宽 3~4)
            draw.rectangle([xmin, ymin, xmax, ymax], outline=color, width=3)

            # 2. 绘制信息背景条与文本
            text = f"{label} [{level}]"
            try:
                text_bbox = draw.textbbox((xmin, max(0, ymin - 22)), text, font=font)
                draw.rectangle(text_bbox, fill=color)
                draw.text((xmin + 2, max(0, ymin - 22)), text, fill=(255, 255, 255), font=font)
            except Exception:
                # 兼容旧版本 PIL
                draw.text((xmin, max(0, ymin - 18)), text, fill=color)

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        img.save(output_path, quality=95)
        return output_path
