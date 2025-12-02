from setuptools import setup, find_packages

# 从 README.md 读取长描述
with open('README.md', 'r', encoding='utf-8') as f:
    long_description = f.read()

# 定义严格的依赖版本约束
install_requires = [
    # 核心深度学习框架
    'torch>=2.0.0,<2.3.0',
    'torchvision>=0.15.0,<0.18.0',
    
    # Transformers 生态系统（严格版本控制）
    'transformers>=4.30.0,<4.40.0',
    'tokenizers>=0.14,<0.19',
    'peft>=0.4.0,<0.5.0',
    'accelerate>=0.21.0',
    
    # 数据处理（NumPy 1.x 兼容性）
    'numpy>=1.24.0,<2.0.0',
    'scipy>=1.10.0',
    'datasets>=2.12.0',
    'pandas>=2.0.0',
    
    # 机器学习工具
    'scikit-learn>=1.3.0',
    'faiss-cpu>=1.7.4',
    
    # 可视化和监控
    'matplotlib>=3.7.0',
    'seaborn>=0.12.0',
    'tensorboard>=2.13.0',
    'wandb>=0.15.0',
    
    # 配置管理
    'hydra-core>=1.3.0',
    'omegaconf>=2.3.0',
    'pyyaml>=6.0',
    
    # 工具和辅助
    'tqdm>=4.65.0',
    'rich>=13.0.0',
]

setup(
    name='bert-clam',
    version='0.1.0',
    author='Your Name',
    author_email='your.email@example.com',
    description='A PyTorch implementation of Continual Learning with Adaptive Memory (CLAM) for BERT.',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/your-repo/bert-clam',
    packages=find_packages(where='.'),
    install_requires=install_requires,
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    python_requires='>=3.8',
    entry_points={
        'console_scripts': [
            'bert-clam-ablation=scripts.run_ablation_study:main',
        ],
    },
)