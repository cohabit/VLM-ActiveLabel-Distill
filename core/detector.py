import json
import base64
import requests

class VLMDetector:
    """阶段1：利用视觉大模型 (VLM) 进行区域检测与定位 (Localization)"""

    def __init__(self, api_key: str = None, api_url: str = None, model_name: str = "Qwen/Qwen2-VL-7B-Instruct"):
        self.api_key = api_key
        self.api_url = api_url or "https://api.siliconflow.cn/v1/chat/completions"
        self.model_name = model_name

    @staticmethod
    def encode_image(image_path: str) -> str:
        """将本地图片转为 Base64 编码"""
        with open(image_path, "rb") as f:
            return base64.b64encode(f.read()).decode("utf-8")

    def detect_regions(self, image_path: str, mock: bool = False):
        """
        执行目标区域检测，提取物体边界框。
        输出格式：[{"box": [xmin, ymin, xmax, ymax]}]
        """
        if mock or not self.api_key or self.api_key.startswith("sk-xxx"):
            print("  [INFO] [Detector] 运行在 Mock 模式（演示数据）...")
            return [
                {"box": [62, 75, 430, 514]},
                {"box": [450, 120, 580, 480]}
            ]

        base64_img = self.encode_image(image_path)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        prompt = """
        你是一个专业的视觉目标检测系统。
        请检测图片中的所有独立物体或食物目标。
        只需输出它们的绝对像素坐标边界框，不要输出任何类别和多余解释。
        必须严格输出为如下 JSON 数组格式：
        [
          {"box": [xmin, ymin, xmax, ymax]}
        ]
        """
        payload = {
            "model": self.model_name,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                    ]
                }
            ],
            "response_format": {"type": "json_object"}
        }

        try:
            resp = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=35,
                proxies={"http": None, "https": None}
            )
            if resp.status_code == 200:
                raw_text = resp.json()['choices'][0]['message']['content']
                clean_json = raw_text.replace("```json", "").replace("```", "").strip()
                parsed = json.loads(clean_json)
                if isinstance(parsed, dict):
                    if "box" in parsed:
                        raw_list = [parsed]
                    else:
                        raw_list = []
                        for k, v in parsed.items():
                            if isinstance(v, list):
                                raw_list = v
                                break
                elif isinstance(parsed, list):
                    raw_list = parsed
                else:
                    raw_list = []
                # 容错与自适应：部分模型输出 0-1000 归一化坐标，部分输出绝对像素
                from PIL import Image
                try:
                    with Image.open(image_path) as im:
                        w, h = im.size
                except Exception:
                    w, h = 1000, 1000

                clean_detections = []
                for item in raw_list:
                    if not isinstance(item, dict) or "box" not in item:
                        continue
                    box = item["box"]
                    if len(box) == 4:
                        # 若数值处于 0-1000 范围且超过了图像分辨率，则按 1000 比例缩放至真实像素
                        if (box[2] > w or box[3] > h) and max(box) <= 1000:
                            item["box"] = [
                                int(box[0] / 1000.0 * w),
                                int(box[1] / 1000.0 * h),
                                int(box[2] / 1000.0 * w),
                                int(box[3] / 1000.0 * h)
                            ]
                    clean_detections.append(item)
                return clean_detections
            else:
                print(f"  [WARN] Detector API 报错 ({resp.status_code}): {resp.text}")
                return []
        except Exception as e:
            print(f"  [ERROR] Detector 请求发生异常: {e}")
            return []
