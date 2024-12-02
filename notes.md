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
- https://blog.csdn.net/v_JULY_v/article/details/134085503
- https://blog.csdn.net/v_JULY_v/article/details/135072211
- 绝对位置编码，指的是每个位置有一个固定的向量值，该向量值被加在词向量中，参与整个神经网络的向前传播计算。相对位置编码只在需要用到token位置信息的地方才引入位置信息，transformer网络中有两个核心网络层，一个是attention，一个是ffn。ffn是点到点的，与位置不相关，所以只有在attention层中需要用到位置信息。而在我们的直觉中，只有token之间的相对位置才是有意义的，比如同样一个句子的两个单词，当他们出现在文章第一段或者最后一段，他们的绝对位置是不同的，但是不应该去影响它们在相对位置上的关系。
- T5相对位置编码：在query向量与key向量的乘积之上加上一个可学习的标量，标量值只和query和key之间的相对位置远近有关。并不是每个m-n都对应一个不同的标量值，0-1对应一个值，2-5对应一个值。并且有截断，当位置差超过128时，它的标量值就不变了
- rope: 是一种相对位置编码的方法，主要用在attention层中，在把query向量与key向量相乘计算attention score的时候，将不同token的位置信息引入到计算过程中。具体的计算过程实际上是把token向量的各个维度两两分组，每组通过一个特定角度的cos和sin值矩阵进行变换，把这个变换过程投射到二维坐标上，就相当于是对词向量做了一个幅角旋转。那具体旋转的角度是和token相对位置的差值相关的，差的越多，旋转角度越大。因此，在训练过程中，模型能够观察到不同的位置差对应的旋转角度给词向量带来的变化，这种变化是有规律的，模型通过学习这种规律，在遇到更长的位置编码时，能够利用这种规律来感知到token之间的位置关系，并把这种位置关系带入到attention的计算中去。
- 外推性：指的是训练时用较短的文本长度，而在推理时使用更长的文本长度。首先绝对位置编码的外推性差，有两种绝对位置编码，一种是可学习的位置编码，它将每个位置one-hot编码转换成一个向量，这个转换过程是通过一个矩阵来映射，矩阵是通过训练学习的，所以它天然无法外推，因为没法把更长的位置编码到一个事先固定维度的one-hot编码上。还有一种是三角函数的位置编码，理论上它可以将更长的位置信息编码成相应的位置编码向量，但是它的转换公式使得它编码得到的位置向量对应的多维数组缺乏一些直观的规律，对于模型来说，他只能通过拟合来记忆已见过的位置信息，但是遇到更远的位置信息时，它无法推理出一个合理的有规律的位置编码值，因此不具备外推性，当推理长度大于训练长度时，出现ppl爆炸的现象。

4. T5
- 与Transformer结构差异：相对位置编码、无bias的LayerNorm、LayerNorm置于残差之外
- 位置编码：头间各异、层间共享的相对位置编码；最大截断长度128

5. Agent构建
- 当前的agent平台提供了单prompt的自动构建功能，实际效果可能不如有一定经验的人手写prompt
- 未来的agent构建是需要不同角色的联结，即多agent构建。达到两个效果：构建不同的agent角色和他们的联结逻辑；允许用户修改单个角色的职责功能，并自动进行上下游角色修正

6. DPO vs PPO
- https://mp.weixin.qq.com/s/mhPJzhQvPJlAWsO2nW9BHg
- 从模型输入输出讲ppo实现的：https://www.cnblogs.com/jiangxinyang/p/17553815.html
- 优势函数的构造：首先是reward，只有在最后一个token有值，根据reward模型输出得到，其余位置是0。然后是kl散度，每个位置上token在此表上的概率分布，通过kl散度约束policy模型与ref模型之间的分布差距，因此reward要减去kl散度。接下来考虑的是通过critic模型预测的价值，这个价值指的是对当前状态能够获得平均收益的预估，因此在每个位置上，它的优势函数还需要加入两个部分，一个是它采取下个动作能够获得的预期收益，然后减去它当前状态能够获得的平均收益，这个差值就能衡量出当前采取策略获得的收益相较于平均收益的增量，然后把这个值加在reward上，就是优势函数。对于

7. Quantization & Overflow
- https://blog.csdn.net/qq_43799400/article/details/134182459
- https://zhuanlan.zhihu.com/p/657886517?theme=dark
- fp32(1+8+23), fp16(1+5+10, 精度0.001), bf16(1+8+7, 精度0.01)

8. GPT-o1
- 强化学习（Let's verify step by step），结果监督（通过针对数学、变成问题的编译或验证器）模型、中间步骤监督模型
- 推理过程的隐藏CoT
- 在普通的文本编辑、撰写、语言风格转换上劣于GPT-o模型

9. PreNorm vs PostNorm
- PostNorm是在残差之后进行归一化，模型的鲁棒性更强
- PreNorm不是所有参数都被正则化，有shortcut（residual connection），整体更不容易发生梯度消失的问题
- 对于浅层模型，梯度消失问题并不大，因此采用PostNorm，而对于深度模型例如llama，则是采用PreNorm

10. distributed training
- https://shivambharuka.medium.com/deep-learning-a-primer-on-distributed-training-part-1-d0ae0054bb1c
- All Reduce = Reduce Scatter + All Gather: 以Ring算法为例，每个阶段都有N-1个传递步骤，在Reduce Scatter阶段完成后，每个节点能拿到某一个chunk的完整数据，在All Gather阶段完成后，所有节点都能拿到所有的完成chunk。在分布式训练过程中，有基于gradient和基于parameter的All Reduce算法，前者是在optimizer步骤之前，在所有节点之间同步梯度值，后者是在每个训练步骤完成后，在所有节点之间同步模型参数。基于gradient的All Reduce是data parallel中常用的方法。
- Zero. 每个节点只保存1/N的optimizer状态，并只更新相应的1/N参数。因此在梯度同步阶段，只执行reduce scatter，执行完之后，每个节点就拥有它那部分参数的所有optimizer状态，因此可以通过更新获得该部分参数的最终更新值。最后在All gather阶段，所有节点都能获得完整的参数。
- activation checkpoints. 对神经网络进行分层，仅保留层间的activation用于重新计算前向
- offloading: CPU, DRAMs and SSDs可以用于offloading
- Overlapping network and compute.
- sequence 并行是什么？
- Megatron PTD-P: TP计算并行利用了连续两个矩阵乘法时，可以先列切再行切，使得卡间通信可以减少3次（一次中间切分的前后通信，和一次中间的all gather），例如MLP和Dropout(Self_attention())操作可以采用

11. Moe
- 负载均衡问题：训练过程中，会倾向于选择相同的几个专家，专家被训练的越多，效果越好，越容易被选择，自我强化。为了保证每个专家收到相同数量的样本，会引入辅助损失（aux_loss），此外还有别的方法：
  - 随机采样：例如在Top-2专家配置中，第一个选择排名最高的，第二个则根据权重比例来随机采样。随机采样可通过增加噪声的方式实现
  - 专家容量：
- 引入 dropout 可以提高稳定性，但会导致模型质量下降。另一方面，增加更多的乘法模块可以提高质量，但会降低模型稳定性。ST-MoE 引入的 Router z-loss 在保持了模型性能的同时显著提升了训练的稳定性
- 稀疏模型更容易出现过拟合，模型内部须加强正则化，可以让系数层有更高的dropout概率
- 在相同的预训练困惑度下，稀疏模型在下游任务中的表现不如对应的稠密模型
- 仅冻结 MoE 层的参数。实验结果显示，这种方法几乎与更新所有参数的效果相当。这种做法可以加速微调过程，并降低显存需求。
- MoE 模型可能从指令式微调中获益更多，甚至超过了稠密模型。此外，MoE 在多任务学习中表现更佳。

12. 关于sft训练
- 数量样本需考虑多样性、一致性和质量，1w左右足够。
    - 多样性：指令多样（不同难度、表达方式）、内容多样（主题、风格、长短）
    - 质量：准确、完备、准确清晰
    - 一致性：
      - 内部：模型自身面对同一任务，在不同语境下是否给出一致的回复
      - 外部：回复是否与外部知识对齐
- 当目标领域与pretrain样本差异过大，需要continue pretrain，为了让模型获得领域知识。如果目标任务与通用任务有较大关联，使用二阶段srf，即混合通用任务与目标任务进行训练。
- 选择chat还是base模型：
  - 资源足够，需同时考虑通用与目标任务，选择base + continue pretrain + 多任务srf
  - 资源足够，只考虑目标领域任务，选择base + continue pretrain + 目标任务srf
  - 资源不够，选择chat + continue pretrain + 目标任务srf
- 训练超参：
  - 学习率，srf为pretrain的十分之一，10^-5左右。不收敛可以考虑降低学习率
  - warmup-ratio，学习率越大，需要设置该值越大。srf通常样本较少，ratio也适当减少
  - epoch，通常需根据loss收敛情况来设置，过少欠拟合，过多过拟合，10w样本1-2个epoch即可收敛
- srf过程中loss会先升后降：可以理解模型在接触到了新的数据分布时，逐渐调整参数过程。
