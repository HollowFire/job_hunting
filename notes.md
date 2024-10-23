1. 为什么decoder-only好？
- 双向注意力容易退化为低秩状态，causal attention必是满秩
- decoder-only架构中每个位置所见信息更少，预测难度更大，在大模型、多数据的情况下，有利于学习更通用的表征
- causal attention具有隐式位置编码

2. PEFT方法分类
- 增量参数微调：adapter、软提示微调
- 选择参数微调：结构化、非结构化
- 重构参数微调：LoRA
- 上述组合

3. 位置编码
- 在sinusoid位置编码中，模型所见的位置编码信息表现为多维数组而非公式中有规律的位置变化，导致其无法捕获位置变化规律，从而对不同的位置信息进行记忆，而非进行编码，论文证明，当推理长度大于训练长度时，模型ppl出现爆炸的现象

4. T5
- 与Transformer结构差异：相对位置编码、无bias的LayerNorm、LayerNorm置于残差之外
- 位置编码：头间各异、层间共享的相对位置编码；最大截断长度128

5. Agent构建
- 当前的agent平台提供了单prompt的自动构建功能，实际效果可能不如有一定经验的人手写prompt
- 未来的agent构建是需要不同角色的联结，即多agent构建。达到两个效果：构建不同的agent角色和他们的联结逻辑；允许用户修改单个角色的职责功能，并自动进行上下游角色修正

6. DPO vs PPO
- https://mp.weixin.qq.com/s/mhPJzhQvPJlAWsO2nW9BHg

7. Quantization & Overflow
- https://blog.csdn.net/qq_43799400/article/details/134182459
- https://zhuanlan.zhihu.com/p/657886517?theme=dark
- fp32(1+8+23), fp16(1+5+10, 精度0.001), bf16(1+8+7, 精度0.01)

8. GPT-o1
- 强化学习（Let's verify step by step），结果监督（通过针对数学、变成问题的编译或验证器）模型、中间步骤监督模型
- 推理过程的隐藏CoT
- 在普通的文本编辑、撰写、语言风格转换上劣于GPT-o模型
