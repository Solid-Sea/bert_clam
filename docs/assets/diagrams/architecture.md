# BERT-CLAM 核心架构图

```mermaid
graph TB
    subgraph "BERT-CLAM Framework"
        Model[BERTCLAMModel<br/>核心模型]
        Manager[StrategyManager<br/>策略管理器]
        
        subgraph "Continual Learning Strategies"
            EWC[EWCStrategy<br/>弹性权重巩固]
            ALP[ALPStrategy<br/>注意力级别保护]
            AMR[AMRStrategy<br/>自适应记忆重放]
            Grammar[GrammarStrategy<br/>语法约束]
        end
        
        subgraph "Base Components"
            BERT[BERT Backbone<br/>预训练模型]
            LoRA[LoRA Adapter<br/>低秩适配器]
        end
    end
    
    Model -->|管理| Manager
    Manager -->|协调| EWC
    Manager -->|协调| ALP
    Manager -->|协调| AMR
    Manager -->|协调| Grammar
    
    Model -->|使用| BERT
    Model -->|集成| LoRA
    
    EWC -->|计算损失| Model
    ALP -->|保护注意力| Model
    AMR -->|记忆重放| Model
    Grammar -->|约束输出| Model
    
    style Model fill:#e1f5ff
    style Manager fill:#fff4e1
    style EWC fill:#e8f5e9
    style ALP fill:#e8f5e9
    style AMR fill:#e8f5e9
    style Grammar fill:#e8f5e9