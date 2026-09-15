# ==============================================================================
# 硅基流动 (SiliconFlow) 配置模板 (示例文件)
# 使用说明：复制本文件为 config.py 并填入您自己的真实 API Key
# ==============================================================================

# 1. 填入你在硅基流动官网申请的 API KEY
# 获取地址：https://cloud.siliconflow.cn/account/ak
SILICONFLOW_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx"

# 2. 硅基流动标准 OpenAI 兼容接口地址
API_URL = "https://api.siliconflow.cn/v1/chat/completions"

# 3. 模型选型配置 (推荐配置硅基流动托管的 Qwen3-VL 系列多模态模型)
MODEL_DETECTOR = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_A = "Qwen/Qwen3-VL-8B-Instruct"
MODEL_B = "Qwen/Qwen3-VL-30B-A3B-Instruct"
MODEL_JUDGE = "Qwen/Qwen3-VL-32B-Instruct"
