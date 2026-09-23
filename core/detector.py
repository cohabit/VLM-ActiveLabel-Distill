import json
import base64
import requests

class VLMDetector:
    """阶段1：利用视觉大模型 (VLM) 进行区域检测与定位 (Localization)"""

    def __init__(self, api_key: str = None, api_url: str = None, model_name: str = "qwen3-vl-plus"):
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
                {"box": [62, 75, 430, 514], "label": "汉堡", "source": self.model_name},
                {"box": [450, 120, 580, 480], "label": "饮料", "source": self.model_name}
            ]

        base64_img = self.encode_image(image_path)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        prompt = """
        你是一个目标定位模块。请检测图片中的独立物体，并输出严格 JSON：
        {"detections": [{"box": [xmin, ymin, xmax, ymax], "label": "初步类别"}]}
        坐标使用图片绝对像素；无法判断类别时填 Unknown。不要输出解释。
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
                    if isinstance(parsed.get("detections"), list):
                        raw_list = parsed["detections"]
                    elif "box" in parsed:
                        raw_list = [parsed]
                    else:
                        raw_list = []
                        for value in parsed.values():
                            if isinstance(value, list):
                                raw_list = value
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
                    if len(box) == 4 and all(isinstance(v, (int, float)) for v in box):
                        # 若数值处于 0-1000 范围且超过了图像分辨率，则按 1000 比例缩放至真实像素
                        if (box[2] > w or box[3] > h) and max(box) <= 1000:
                            item["box"] = [
                                int(box[0] / 1000.0 * w),
                                int(box[1] / 1000.0 * h),
                                int(box[2] / 1000.0 * w),
                                int(box[3] / 1000.0 * h)
                            ]
                        item["box"] = [
                            max(0, min(w - 1, int(item["box"][0]))),
                            max(0, min(h - 1, int(item["box"][1]))),
                            max(0, min(w - 1, int(item["box"][2]))),
                            max(0, min(h - 1, int(item["box"][3])))
                        ]
                        if item["box"][2] > item["box"][0] and item["box"][3] > item["box"][1]:
                            item["source"] = self.model_name
                            clean_detections.append(item)
                return clean_detections
            else:
                print(f"  [WARN] Detector API 报错 ({resp.status_code}): {resp.text}")
                return []
        except Exception as e:
            print(f"  [ERROR] Detector 请求发生异常: {e}")
            return []
