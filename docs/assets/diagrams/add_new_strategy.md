# 如何添加一个新策略

```mermaid
graph TD
    Start([💡 提出新策略想法]) --> Design[📐 设计策略算法]
    Design --> Create[📝 创建策略类文件]
    Create --> Inherit[🔗 继承 BaseStrategy]
    
    Inherit --> Implement{实现核心方法}
    Implement -->|必需| Init[__init__<br/>初始化参数]
    Implement -->|必需| Compute[compute_loss<br/>计算损失]
    Implement -->|可选| Update[update_after_task<br/>任务后更新]
    
    Init --> Register[📋 在 StrategyManager 中注册]
    Compute --> Register
    Update --> Register
    
    Register --> Config[⚙️ 在配置文件中启用]
    Config --> Test[🧪 编写单元测试]
    Test --> Validate{验证效果}
    
    Validate -->|通过| Document[📚 更新文档]
    Validate -->|失败| Debug[🔧 调试优化]
    Debug --> Test
    
    Document --> Success([✅ 成功集成])
    
    style Start fill:#e1f5ff
    style Success fill:#e8f5e9
    style Validate fill:#fff4e1
    style Debug fill:#ffebee