# ==============================================================================
# 硅基流动 (SiliconFlow) 配置模板 (示例文件)
# 使用说明：复制本文件为 config.py 并填入您自己的真实 API Key
# ==============================================================================

# 1. 填入你在硅基流动官网申请的 API KEY
# 获取地址：https://cloud.siliconflow.cn/account/ak
SILICONFLOW_API_KEY = "sk-xxxxxxxxxxxxxxxxxxxxxxxx"

# 2. 硅基流动标准 OpenAI 兼容接口地址
API_URL = "https://api.siliconflow.cn/v1/chat/completions"

# 3. 项目中各模型角色。不同厂商的 OpenAI 兼容网关可分别配置。
# 这里保留与简历/面试口径一致的角色名；默认 demo 不调用外部 API。
MODEL_DETECTOR = "qwen3-vl-plus"
MODEL_SECOND_VISION = "Doubao-seed-1-6-vision"
MODEL_LOW_COST_REVIEW = "glm-4.6v-flashx"
MODEL_HARD_CASE_REVIEW = "kimi-k2.6"
MODEL_STRUCTURED_ROUTER = "deepseek-v4-flash"
