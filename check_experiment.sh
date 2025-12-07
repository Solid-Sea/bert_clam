#!/bin/bash
# 检查远程服务器上的实验进度

echo "=== 检查 tmux 会话状态 ==="
wsl sshpass -p 'zt7MYfXmYlpD' ssh -p 33183 root@connect.westc.gpuhub.com "tmux ls"

echo ""
echo "=== 最近的训练输出 ==="
wsl sshpass -p 'zt7MYfXmYlpD' ssh -p 33183 root@connect.westc.gpuhub.com "tmux capture-pane -t bert_clam -p | tail -50"

echo ""
echo "=== 检查是否有结果文件 ==="
wsl sshpass -p 'zt7MYfXmYlpD' ssh -p 33183 root@connect.westc.gpuhub.com "ls -lh /root/autodl-tmp/bert_clam/experiments/ 2>/dev/null || echo '结果目录尚未创建'"