1. 为什么decoder-only好？
- 双向注意力容易退化为低秩状态，causal attention必是满秩
- decoder-only架构中每个位置所见信息更少，预测难度更大，在大模型、多数据的情况下，有利于学习更通用的表征
- causal attention具有隐式位置编码

2. PEFT方法分类
- 增量参数微调：adapter、软提示微调
- 选择参数微调：结构化、非结构化
- 重构参数微调：LoRA
- 上述组合
