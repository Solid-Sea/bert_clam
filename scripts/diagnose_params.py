import torch
from bert_clam.models.bert_clam_model import BERTCLAMModel

# 测试配置
model_config = {
    "enable_ewc": True,
    "ewc_lambda": 1000.0,
    "enable_amr": False,
    "enable_alp": True,
    "alp_top_k": 3,
    "enable_grammar": False,
    "lora_enabled": False
}

print("创建模型...")
model = BERTCLAMModel(
    model_name="bert-base-uncased",
    num_labels=3,
    enable_ewc=model_config.get("enable_ewc", False),
    enable_amr=model_config.get("enable_amr", False),
    enable_alp=model_config.get("enable_alp", False),
    enable_grammar=model_config.get("enable_grammar", False),
    ewc_lambda=model_config.get("ewc_lambda", 0.0),
    amr_k=model_config.get("amr_k", 5),
    alp_top_k=model_config.get("alp_top_k", 3),
    lora_enabled=model_config.get("lora_enabled", False)
)

print("\n检查参数状态:")
total_params = 0
trainable_params = 0
frozen_params = 0

for name, param in model.named_parameters():
    total_params += param.numel()
    if param.requires_grad:
        trainable_params += param.numel()
    else:
        frozen_params += param.numel()
        print(f"  冻结: {name}, shape={param.shape}")

print(f"\n总参数: {total_params}")
print(f"可训练参数: {trainable_params}")
print(f"冻结参数: {frozen_params}")

# 测试优化器
trainable_param_list = [p for p in model.parameters() if p.requires_grad]
print(f"\n优化器会找到 {len(trainable_param_list)} 个可训练参数张量")
print(f"优化器会优化 {sum(p.numel() for p in trainable_param_list)} 个参数")