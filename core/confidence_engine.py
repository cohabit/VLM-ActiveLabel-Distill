class ConfidenceEngine:
    """阶段3：四级置信度决策引擎与人机协同 (Human-in-the-Loop) 分流"""

    LEVEL_1_CONSENSUS = "Level-1: Direct-Adopt (Consensus)"
    LEVEL_2_JUDGE_HIGH = "Level-2: Direct-Adopt (Judge Confident)"
    LEVEL_3_HUMAN_REVIEW = "Level-3: Human-Review (Hard Example)"
    LEVEL_4_DISCARD = "Level-4: Discard (Noise/Invalid)"

    @classmethod
    def evaluate(cls, item_result: dict) -> tuple:
        """
        根据多模型比对和裁判结果，评估置信度级别与路由方向。
        返回: (level_name, action) -> action 为 'AUTO_ADOPT', 'HUMAN_REVIEW', 'DISCARD'
        """
        if not item_result:
            return cls.LEVEL_4_DISCARD, "DISCARD"

        # 1. 如果多模型完全一致
        if item_result.get("consensus") is True:
            return cls.LEVEL_1_CONSENSUS, "AUTO_ADOPT"

        # 2. 如果存在分歧，参考裁判模型打分
        judge_res = item_result.get("judge_result")
        if judge_res:
            score = judge_res.get("confidence_score", 0)
            if score >= 85:
                return cls.LEVEL_2_JUDGE_HIGH, "AUTO_ADOPT"
            else:
                return cls.LEVEL_3_HUMAN_REVIEW, "HUMAN_REVIEW"

        return cls.LEVEL_3_HUMAN_REVIEW, "HUMAN_REVIEW"

    @staticmethod
    def export_to_label_studio_task(image_url_or_path: str, box: list, label: str, image_width: int, image_height: int) -> dict:
        """
        将异常样本组装为 Label Studio 平台标准的预标注任务格式 (Task Schema)。
        注意：必须将绝对像素坐标归一化为百分比 (0-100)。
        """
        xmin, ymin, xmax, ymax = box
        
        # 归一化为百分比
        x_pct = (xmin / image_width) * 100.0
        y_pct = (ymin / image_height) * 100.0
        w_pct = ((xmax - xmin) / image_width) * 100.0
        h_pct = ((ymax - ymin) / image_height) * 100.0

        return {
            "data": {
                "image": image_url_or_path
            },
            "predictions": [
                {
                    "model_version": "vlm_ensemble_v1",
                    "result": [
                        {
                            "type": "rectanglelabels",
                            "value": {
                                "x": round(x_pct, 2),
                                "y": round(y_pct, 2),
                                "width": round(w_pct, 2),
                                "height": round(h_pct, 2),
                                "rectanglelabels": [label]
                            }
                        }
                    ]
                }
            ]
        }
