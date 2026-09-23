import base64
import io
import json
import re
from typing import Any, Dict, Optional

import requests
from PIL import Image

class EnsembleJudgeClassifier:
    """多模型视觉工作流。

    qwen3-vl-plus 负责原图初筛（由 detector 完成）；本模块负责第二路视觉
    识别、低成本复核、疑难升级，以及文本结构化汇总。最终可靠度由
    ``ConfidenceEngine`` 根据模型一致性和确定性质量信号计算，不采用模型自报分数。
    """

    def __init__(self, api_key: str = None, api_url: str = None,
                 model_a: str = "Doubao-seed-1-6-vision",
                 model_b: str = "glm-4.6v-flashx",
                 model_judge: str = "deepseek-v4-flash",
                 model_hard_case: str = "kimi-k2.6",
                 model_detector: str = "qwen3-vl-plus"):
        self.api_key = api_key
        self.api_url = api_url or "https://api.siliconflow.cn/v1/chat/completions"
        self.model_a = model_a
        self.model_b = model_b
        self.model_judge = model_judge
        self.model_hard_case = model_hard_case
        self.model_detector = model_detector

    @staticmethod
    def crop_region_to_base64(image: Image.Image, box: list) -> str:
        """根据坐标 [xmin, ymin, xmax, ymax] 裁剪图片并转为 Base64"""
        xmin, ymin, xmax, ymax = box
        cropped = image.crop((xmin, ymin, xmax, ymax))
        buffer = io.BytesIO()
        cropped.save(buffer, format="JPEG")
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    def _call_single_model(self, model_name: str, prompt: str, base64_img: Optional[str] = None):
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": model_name,
            "messages": [{
                "role": "user",
                "content": ([{"type": "text", "text": prompt}] + ([
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_img}"}}
                ] if base64_img else []))
            }],
            "response_format": {"type": "json_object"}
        }
        try:
            resp = requests.post(
                self.api_url,
                headers=headers,
                json=payload,
                timeout=30,
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

    @staticmethod
    def _canonical_label(label: Any) -> str:
        value = str(label or "Unknown").strip().lower()
        aliases = {"马铃薯": "土豆", "potato": "土豆", "可乐": "饮料", "cola": "饮料"}
        return aliases.get(value, value)

    @classmethod
    def _same_prediction(cls, left: Dict[str, Any], right: Dict[str, Any]) -> bool:
        return cls._canonical_label(left.get("label")) == cls._canonical_label(right.get("label"))

    def _mock_prediction(self, box: list, role: str) -> Dict[str, Any]:
        # 可复现的演示分支，避免随机结果造成报告无法复核。
        x = int(box[0]) if box else 0
        if role == "second_vision":
            return {"label": "汉堡" if x < 300 else "饮料", "brand": "Unknown", "source": self.model_a}
        if role == "low_cost_review":
            # 第二个目标故意保留一个分歧，用来演示疑难升级和人工兜底分支。
            return {"label": "汉堡" if x < 300 else "咖啡", "brand": "Unknown", "source": self.model_b}
        if role == "hard_case_review":
            return {"label": "汉堡" if x < 300 else "饮料", "brand": "Unknown", "source": self.model_hard_case}
        return {"label": "Unknown", "brand": "Unknown", "source": self.model_judge}

    def classify_crop(self, original_image_path: str, box: list, mock_mode: bool = False,
                      initial_prediction: Optional[Dict[str, Any]] = None):
        """
        对裁剪区域进行多模型交叉分类与裁判仲裁。
        返回: dict (包含各模型意见、裁判结论、决策依据)
        """
        if mock_mode or not self.api_key or self.api_key.startswith("sk-xxx"):
            initial = initial_prediction or {"label": "Unknown", "source": self.model_detector}
            second = self._mock_prediction(box, "second_vision")
            review = self._mock_prediction(box, "low_cost_review")
            model_results = {self.model_detector: initial, self.model_a: second, self.model_b: review}
            agree = self._same_prediction(initial, second) and self._same_prediction(second, review)
            hard = None if agree else self._mock_prediction(box, "hard_case_review")
            final = hard or second
            return {
                "model_results": model_results,
                "consensus": agree,
                "review_path": [self.model_a, self.model_b] + ([self.model_hard_case] if hard else []),
                "hard_case_review": hard,
                "structured_router": {"route": "AUTO" if agree else "REVIEW", "source": self.model_judge},
                "judge_needed": not agree,
                "judge_result": hard,
                "final_label": final.get("label", "Unknown"),
                "final_brand": final.get("brand", "Unknown")
            }

        # 真实 API 模式
        img = Image.open(original_image_path)
        crop_b64 = self.crop_region_to_base64(img, box)

        classify_prompt = "输出严格 JSON：{\"label\":\"类别\",\"brand\":\"品牌或 Unknown\"}。只识别局部图像中的目标。"
        initial = initial_prediction or self._call_single_model(self.model_detector, classify_prompt, crop_b64) or {"label": "Unknown"}
        res_a = self._call_single_model(self.model_a, classify_prompt, crop_b64) or {"label": "Unknown"}
        res_b = self._call_single_model(self.model_b, classify_prompt, crop_b64) or {"label": "Unknown"}
        model_results = {self.model_detector: initial, self.model_a: res_a, self.model_b: res_b}
        consensus = self._same_prediction(initial, res_a) and self._same_prediction(res_a, res_b)
        hard_review = None
        if not consensus:
            hard_review = self._call_single_model(
                self.model_hard_case,
                "请复核候选结果并输出 {\"label\":\"类别\",\"brand\":\"品牌或 Unknown\"}，无法确定时 label 填 Unknown。候选：" + json.dumps(model_results, ensure_ascii=False),
                crop_b64,
            )
        final = hard_review or res_a
        router_prompt = "根据以下视觉结果输出严格 JSON：{\"route\":\"AUTO或REVIEW\",\"reason\":\"简短原因\"}。不要生成新的视觉结论。\n" + json.dumps({"results": model_results, "hard_review": hard_review}, ensure_ascii=False)
        router = self._call_single_model(self.model_judge, router_prompt) or {"route": "AUTO" if consensus else "REVIEW"}
        return {
            "model_results": model_results,
            "consensus": consensus,
            "review_path": [self.model_a, self.model_b] + ([self.model_hard_case] if hard_review else []),
            "hard_case_review": hard_review,
            "structured_router": router,
            "judge_needed": not consensus,
            "judge_result": hard_review,
            "final_label": final.get("label", "Unknown"),
            "final_brand": final.get("brand", "Unknown")
        }
