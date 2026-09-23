import os
import sys
import glob
import argparse
from PIL import Image

# 禁用代理，防止国内 API (api.siliconflow.cn) 被代理软件拦截
for proxy_var in ["http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"]:
    os.environ.pop(proxy_var, None)
os.environ["NO_PROXY"] = "*"

# 解决 Windows 终端编码问题
if sys.platform.startswith("win"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# 导入配置
try:
    import config as user_config
    SILICONFLOW_API_KEY = getattr(user_config, "SILICONFLOW_API_KEY", "")
    API_URL = getattr(user_config, "API_URL", "https://api.siliconflow.cn/v1/chat/completions")
except ImportError:
    SILICONFLOW_API_KEY = "sk-xxxxxxxx"
    API_URL = "https://api.siliconflow.cn/v1/chat/completions"

# Keep the public demo configuration aligned with the current architecture.
# API credentials and endpoints may come from config.py, but fixed role names
# prevent an old local config from silently reintroducing stale model names.
MODEL_DETECTOR = "qwen3-vl-plus"
MODEL_SECOND_VISION = "Doubao-seed-1-6-vision"
MODEL_LOW_COST_REVIEW = "glm-4.6v-flashx"
MODEL_HARD_CASE_REVIEW = "kimi-k2.6"
MODEL_STRUCTURED_ROUTER = "deepseek-v4-flash"

# 导入流水线模块
from core.detector import VLMDetector
from core.classifier import EnsembleJudgeClassifier
from core.confidence_engine import ConfidenceEngine
from core.yolo_exporter import YOLOExporter
from core.visualizer import BoundingBoxVisualizer

def process_single_image(image_path: str, detector, classifier, exporter, is_mock: bool):
    """端到端处理单张图片，完成4个阶段并输出可视化标注"""
    base_name = os.path.basename(image_path)
    file_id, _ = os.path.splitext(base_name)

    img = Image.open(image_path)
    img_width, img_height = img.size

    print("-" * 60)
    print(f">> 开始处理图像: {base_name} ({img_width}x{img_height})")

    # ---------------- 阶段 1：区域检测 ----------------
    detections = detector.detect_regions(image_path, mock=is_mock)
    print(f"  [阶段1] 检测完成，发现 {len(detections)} 个目标候选框")

    # ---------------- 阶段 2：裁剪与多模型分类 ----------------
    classified_items = []
    for idx, d in enumerate(detections):
        box = d["box"]
        c_res = classifier.classify_crop(
            image_path,
            box,
            mock_mode=is_mock,
            initial_prediction={"label": d.get("label", "Unknown"), "source": MODEL_DETECTOR},
        )
        c_res["box"] = box
        label = c_res.get("final_label", "Unknown")
        brand = c_res.get("final_brand", "Unknown")
        
        if c_res.get("consensus"):
            print(f"    - 目标 {idx+1} {box}: [一致] '{label}' (品牌: {brand})")
        else:
            print(f"    - 目标 {idx+1} {box}: [升级复核] '{label}' (路径: {c_res.get('review_path', [])})")
        classified_items.append(c_res)

    # ---------------- 阶段 3：四级置信度分层 ----------------
    auto_adopt_queue = []
    human_review_queue = []
    visualize_items = []

    for item in classified_items:
        level, action = ConfidenceEngine.evaluate(item, image_size=(img_width, img_height))
        item["action"] = action
        item["level"] = level
        
        visualize_items.append({
            "box": item["box"],
            "label": item.get("final_label", "Unknown"),
            "action": action,
            "level": level.split(":")[0],
            "reliability_score": item.get("reliability_score", 0.0)
        })

        if action == "AUTO_ADOPT":
            auto_adopt_queue.append(item)
        elif action == "HUMAN_REVIEW":
            ls_task = ConfidenceEngine.export_to_label_studio_task(
                image_url_or_path=image_path,
                box=item["box"],
                label=item["final_label"],
                image_width=img_width,
                image_height=img_height
            )
            human_review_queue.append(ls_task)

    # ---------------- 阶段 4：导出 YOLO 与生成可视化结果 ----------------
    # 1. 导出 YOLO 标注文件
    yolo_lines = []
    for item in auto_adopt_queue:
        line = exporter.convert_to_yolo_line(
            box=item["box"],
            label=item["final_label"],
            img_width=img_width,
            img_height=img_height
        )
        yolo_lines.append(line)

    label_txt_path = f"runtime_output/labels/{file_id}.txt"
    exporter.save_yolo_annotation(label_txt_path, yolo_lines)

    # 2. 绘制可视化边界框 (可直观检查位置是否准确)
    vis_output_path = f"runtime_output/visualized/{base_name}"
    BoundingBoxVisualizer.draw_annotations(
        image_path=image_path,
        annotated_items=visualize_items,
        output_path=vis_output_path
    )

    print(f"  [阶段3/4] 完成！自动采纳: {len(auto_adopt_queue)} 个，待人工复核: {len(human_review_queue)} 个")
    print(f"    * YOLO 标签已生成: {label_txt_path}")
    print(f"    * 可视化画框已生成: {vis_output_path} (可直接打开验框!)")
    return {
        "file": base_name,
        "total": len(detections),
        "auto": len(auto_adopt_queue),
        "review": len(human_review_queue),
        "vis_path": vis_output_path
    }

def main():
    parser = argparse.ArgumentParser(description="VLM-ActiveLabel-Distill 批量多图像标注与蒸馏流水线")
    parser.add_argument("--input-dir", type=str, default="data/inputs", help="测试图片所在文件夹")
    parser.add_argument("--api-key", type=str, default=SILICONFLOW_API_KEY, help="API Key (默认读取 config.py)")
    parser.add_argument("--mock", action="store_true", help="强制以模拟数据模式运行")
    args = parser.parse_args()

    is_real_key = args.api_key and not args.api_key.startswith("sk-xxx") and len(args.api_key) > 10
    is_mock = args.mock or (not is_real_key)

    # 获取输入文件夹下的所有图片
    os.makedirs(args.input_dir, exist_ok=True)
    image_files = sorted(
        glob.glob(os.path.join(args.input_dir, "*.jpg")) +
        glob.glob(os.path.join(args.input_dir, "*.png"))
    )

    if not image_files:
        print(f"[WARN] 文件夹 {args.input_dir} 中暂无图片，正在生成默认测试图片...")
        default_img = os.path.join(args.input_dir, "01_sample.jpg")
        Image.new("RGB", (600, 480), color=(240, 240, 240)).save(default_img)
        image_files = [default_img]

    print("=" * 65)
    print(">> 启动【VLM-ActiveLabel-Distill】批量多模态智能标注与蒸馏流水线")
    print(f">> 输入目录: {os.path.abspath(args.input_dir)} (共 {len(image_files)} 张图片)")
    print(f">> 运行模式: {'[Mock 演示模式]' if is_mock else '[硅基流动真实 Qwen3-VL 模式]'}")
    print(f">> 可视化输出目录: runtime_output/visualized/")
    print(f">> YOLO 标注输出目录: runtime_output/labels/")
    print("=" * 65)

    # 初始化四大模块
    detector = VLMDetector(api_key=args.api_key, api_url=API_URL, model_name=MODEL_DETECTOR)
    classifier = EnsembleJudgeClassifier(
        api_key=args.api_key,
        api_url=API_URL,
        model_a=MODEL_SECOND_VISION,
        model_b=MODEL_LOW_COST_REVIEW,
        model_hard_case=MODEL_HARD_CASE_REVIEW,
        model_judge=MODEL_STRUCTURED_ROUTER,
        model_detector=MODEL_DETECTOR,
    )
    exporter = YOLOExporter()

    summaries = []
    all_items = []
    for img_path in image_files:
        res = process_single_image(img_path, detector, classifier, exporter, is_mock)
        summaries.append(res)
        all_items.append(res)

    # 生成全局 YOLO dataset.yaml
    output_yaml = "runtime_output/dataset.yaml"
    exporter.generate_dataset_yaml(dataset_dir=os.path.abspath("runtime_output"), output_yaml_path=output_yaml)

    # 打印全局汇总看板
    print("\n" + "=" * 65)
    print("📊 批量标注与模型蒸馏全流程完成看板 (Batch Summary)")
    print(f"{'图像文件名':<20} | {'检测目标数':<10} | {'自动采纳':<10} | {'人工复核':<10}")
    print("-" * 65)
    for s in summaries:
        print(f"{s['file']:<20} | {s['total']:<10} | {s['auto']:<14} | {s['review']:<12}")
    print("=" * 65)
    print("💡 验框提示: 请直接打开 [runtime_output/visualized/] 文件夹，检查框位置和分流等级。")

    report_path = os.path.join("runtime_output", "annotation_run_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        import json
        json.dump({
            "project": "VLM-ActiveLabel-Distill",
            "mode": "mock" if is_mock else "api",
            "model_roles": {
                "detector": MODEL_DETECTOR,
                "second_vision": MODEL_SECOND_VISION,
                "low_cost_review": MODEL_LOW_COST_REVIEW,
                "hard_case_review": MODEL_HARD_CASE_REVIEW,
                "structured_router": MODEL_STRUCTURED_ROUTER,
            },
            "summaries": summaries,
            "disclaimer": "本报告基于脱敏/模拟输入，仅用于验证流程，不代表生产环境指标。",
        }, f, ensure_ascii=False, indent=2)
    print(f"💡 运行审计报告: {report_path}")

if __name__ == "__main__":
    main()
