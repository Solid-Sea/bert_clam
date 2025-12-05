"""测试Notebook代码的可执行性"""
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from transformers import BertTokenizer

from bert_clam.models.bert_clam_model import BERTCLAMModel
from bert_clam.core.ewc import EnhancedElasticWeightConsolidation
from bert_clam.core.memory_replay_bank import EnhancedAdaptiveMemoryRetrieval
from bert_clam.core.grammar_aware import EnhancedGrammarAwareModule
from bert_clam.core.strategy import EWCStrategy, MRBStrategy, GrammarStrategy

def create_dummy_data(tokenizer, num_samples=16, max_length=128):
    """创建虚拟数据"""
    texts = [
        "This is a positive example.",
        "This is a negative example.",
        "Another positive sentence here.",
        "Yet another negative sentence."
    ] * (num_samples // 4)
    
    labels = [1, 0, 1, 0] * (num_samples // 4)
    
    encodings = tokenizer(
        texts,
        padding='max_length',
        truncation=True,
        max_length=max_length,
        return_tensors='pt'
    )
    
    dataset = []
    for i in range(len(texts)):
        dataset.append({
            'input_ids': encodings['input_ids'][i],
            'attention_mask': encodings['attention_mask'][i],
            'labels': torch.tensor(labels[i])
        })
    
    return DataLoader(dataset, batch_size=8, shuffle=True)

def main():
    print("=== 测试Notebook代码 ===\n")
    
    # 1. 初始化
    print("1. 初始化tokenizer和设备...")
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"   设备: {device}")
    
    # 2. 创建数据
    print("\n2. 创建虚拟数据...")
    task0_dataloader = create_dummy_data(tokenizer, num_samples=16)
    print(f"   数据批次: {len(task0_dataloader)}")
    
    # 3. 初始化模块
    print("\n3. 初始化核心模块...")
    ewc_module = EnhancedElasticWeightConsolidation(lambda_ewc=0.5, fisher_samples=10)
    mrb_module = EnhancedAdaptiveMemoryRetrieval(hidden_size=768, k=5, memory_dim=768)
    grammar_module = EnhancedGrammarAwareModule(hidden_size=768, num_attention_heads=12, grammar_features_dim=64)
    print("   ✓ 模块初始化完成")
    
    # 4. 创建模型和策略
    print("\n4. 创建模型和策略...")
    model = BERTCLAMModel(
        model_name='bert-base-uncased',
        num_labels=2,
        lora_r=8,
        lora_alpha=16,
        device=device,
        lora_enabled=True,
        enable_ewc=False,
        enable_amr=False,
        enable_grammar=False
    ).to(device)
    
    strategies = [
        GrammarStrategy(grammar_module.to(device)),
        MRBStrategy(mrb_module.to(device), fusion_weight=0.2),
        EWCStrategy(ewc_module, model)
    ]
    model.strategies = strategies
    print(f"   ✓ 策略数量: {len(model.strategies)}")
    
    # 5. 训练测试
    print("\n5. 执行训练测试...")
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    model.train()
    model.register_task(0)
    
    step = 0
    for batch in task0_dataloader:
        if step >= 2:  # 只测试2步
            break
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            task_id=0
        )
        
        loss = outputs['loss']
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        print(f"   步骤 {step+1}: 损失={loss.item():.4f}")
        step += 1
    
    # 6. 推理测试
    print("\n6. 执行推理测试...")
    model.eval()
    with torch.no_grad():
        batch = next(iter(task0_dataloader))
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            task_id=0
        )
        
        predictions = torch.argmax(outputs['logits'], dim=-1)
        print(f"   预测形状: {predictions.shape}")
        print(f"   预测值: {predictions[:4].tolist()}")
    
    print("\n✓ 所有测试通过！Notebook代码可正常运行。")

if __name__ == '__main__':
    main()