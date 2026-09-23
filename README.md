# VLM-ActiveLabel-Distill

脱敏图片驱动的多模态协同标注与端侧训练数据闭环 Demo。它把“候选框生成—第二路视觉识别—低成本复核—疑难升级—结构化分流—YOLO 标签导出”串成可复核的离线流水线。

> 仓库不包含实习项目原始图片、涉密数据或 API Key。默认 `--mock` 运行；真实 API 只通过本地 `config.py` 接入，不能把 Demo 结果当成生产指标。

## 模型角色

| 角色 | 模型 | 任务 |
|---|---|---|
| 原图候选框和初步类别 | `qwen3-vl-plus` | 生成候选框 |
| 第二路视觉识别 | `Doubao-seed-1-6-vision` | 对目标区域切片复核 |
| 低成本视觉复核 | `glm-4.6v-flashx` | 处理分歧样本 |
| 疑难样本升级 | `kimi-k2.6` | 遮挡、细分类混淆复核 |
| 结构化汇总与分流建议 | `deepseek-v4-flash` | 汇总结果、检查规则 |

最终可靠度由确定性引擎根据模型一致性、框合法性、目标面积和复核路径计算，不使用模型自报置信度。

## 处理链路

```text
图片
 -> qwen3-vl-plus 候选框
 -> crop + Doubao / GLM 视觉结果
 -> 坐标匹配 + 同义类别归一
 -> Kimi 疑难升级（必要时）
 -> DeepSeek 结构化汇总
 -> 四级分流：自动通过 / 二次复核 / 待标注 / 人工处理
 -> 可视化结果 + YOLO 标签 + JSON 审计报告
```

## 运行

```bash
pip install -r requirements.txt
python demo.py --mock
```

默认输出：

- `runtime_output/visualized/`：带分流标签的可视化图片；
- `runtime_output/labels/`：自动采纳样本的 YOLO 标签；
- `runtime_output/dataset.yaml`：类别映射；
- `runtime_output/annotation_run_report.json`：模型角色、运行模式和批量汇总。

如需接入 OpenAI 兼容网关，复制 `config.example.py` 为 `config.py` 并填入密钥。各厂商 API 地址和认证方式可能不同，默认代码只保证 Mock 流程可复现。

## 工程边界

- 这是个人脱敏验证 Demo，不是完整的端侧训练平台；YOLO 导出是数据交付接口，仓库不自动训练小模型。
- `ConfidenceEngine` 负责确定性分流；模型只提供候选结论，不直接决定最终可靠度。
- 无法形成稳定结论的样本进入人工复核，不强行投票。
