import io
import json
import base64
import requests
from PIL import Image

class EnsembleJudgeClassifier:
    """阶段2：局部特征裁剪 (Crop) + 多模型背靠背投票 + 裁判机制 (LLM-as-a-Judge)"""

    def __init__(self, api_key: str = None, api_url: str = None,
                 model_a: str = "Qwen/Qwen2-VL-7B-Instruct",
                 model_b: str = "Pro/Qwen/Qwen2-VL-7B-Instruct",
                 model_judge: str = "Qwen/Qwen2-VL-72B-Instruct"):
        self.api_key = api_key
        self.api_url = api_url or "https://api.siliconflow.cn/v1/chat/completions"
        self.model_a = model_a
        self.model_b = model_b
        self.model_judge = model_judge

    @staticmethod
    def crop_region_to_base64(image: Image.Image, box: list) -> str:
        """根据坐标 [xmin, ymin, xmax, ymax] 裁剪图片并转为 Base64"""
        xmin, ymin, xmax, ymax = box
        cropped = image.crop((xmin, ymin, xmax, ymax))
        buffer = io.BytesIO()
        cropped.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _call_single_model(self, model_name: str, prompt: str, base64_img: str):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model_name,
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
                timeout=30,
                proxies={"http": None, "https": None}
            )
            if resp.status_code == 200:
                raw_text = resp.json()['choices'][0]['message']['content']
                clean = raw_text.replace("```json", "").replace("```", "").strip()
                return json.loads(clean)
            else:
                # 打印详细方便排查
                print(f"     [WARN] 模型 {model_name} 响应码: {resp.status_code}")
        except Exception as e:
            print(f"     [ERROR] 模型 {model_name} 请求异常: {e}")
        return None

    def classify_crop(self, original_image_path: str, box: list, mock_mode: bool = False):
        """
        对裁剪区域进行多模型交叉分类与裁判仲裁。
        返回: dict (包含各模型意见、裁判结论、决策依据)
        """
        if mock_mode or not self.api_key or self.api_key.startswith("sk-xxx"):
            import random
            is_conflict = random.random() > 0.4
            if is_conflict:
                return {
                    "model_a": {"label": "汉堡", "brand": "麦当劳"},
                    "model_b": {"label": "汉堡", "brand": "汉堡王"},
                    "consensus": False,
                    "judge_needed": True,
                    "judge_result": {"label": "汉堡", "brand": "麦当劳", "confidence_score": 92},
                    "final_label": "汉堡",
                    "final_brand": "麦当劳"
                }
            else:
                return {
                    "model_a": {"label": "薯条", "brand": "麦当劳"},
                    "model_b": {"label": "薯条", "brand": "麦当劳"},
                    "consensus": True,
                    "judge_needed": False,
                    "judge_result": None,
                    "final_label": "薯条",
                    "final_brand": "麦当劳"
                }

        # 真实 API 模式
        img = Image.open(original_image_path)
        crop_b64 = self.crop_region_to_base64(img, box)

        classify_prompt = """
        请识别这张局部特写图中的目标。输出严格 JSON 格式：
        {
          "label": "物体类别名称（如汉堡、薯条、可乐）",
          "brand": "品牌名称（如不确定填 Unknown）"
        }
        """

        res_a = self._call_single_model(self.model_a, classify_prompt, crop_b64) or {"label": "Unknown", "brand": "Unknown"}
        res_b = self._call_single_model(self.model_b, classify_prompt, crop_b64) or {"label": "Unknown", "brand": "Unknown"}

        label_a = res_a.get("label", "Unknown")
        label_b = res_b.get("label", "Unknown")
        brand_a = res_a.get("brand", "Unknown")
        brand_b = res_b.get("brand", "Unknown")

        if label_a == label_b and brand_a == brand_b:
            return {
                "model_a": res_a,
                "model_b": res_b,
                "consensus": True,
                "judge_needed": False,
                "judge_result": None,
                "final_label": label_a,
                "final_brand": brand_a
            }
        else:
            # 触发裁判
            judge_prompt = f"""
            你作为高级仲裁专家。模型A的判断是: {res_a}；模型B的判断是: {res_b}。
            请仔细观察特写图做出最终仲裁。输出严格 JSON 格式：
            {{
              "label": "最终判定的准确类别",
              "brand": "最终判定的品牌",
              "confidence_score": 0到100的置信度整数
            }}
            """
            judge_res = self._call_single_model(self.model_judge, judge_prompt, crop_b64) or {
                "label": label_a, "brand": brand_a, "confidence_score": 75
            }
            return {
                "model_a": res_a,
                "model_b": res_b,
                "consensus": False,
                "judge_needed": True,
                "judge_result": judge_res,
                "final_label": judge_res.get("label", label_a),
                "final_brand": judge_res.get("brand", brand_a)
            }
