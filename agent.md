1. 目前做的工作
今年初开始做agent开发平台，平台提供的功能主要包括：基本的智能体创建和管理，包括知识库、提示词、工具、记忆这些功能模块，同时也支持workflow工作流，允许用户通过配置多个节点，
实现复杂的工作流程（大模型节点、plugin节点、知识库节点、code节点）。我这边目前负责到的工作，在算法方面主要负责两个部分内容：
一个是在知识库中，针对RAG场景大模型的上下文内容管理需求，做了一个针对网页段落的优选过滤模块。
还有一个是在agent开发平台上，提供提示词的自动创建和自动优化算法。
自动创建这一块，主要研究两类算法，一种是自动化的一键创建，根据用户提供的agent名称或者agent描述，根据元模板来直接生成一个system prompt。这个生成过程的话，主要是识别到用户的目标场景，按照结构化的模板样式，
生成角色描述、技能描述、约束条件和输出格式这样一些具体内容。我们还会在元模板中插入工具信息，例如用户在agent中配置了订机票酒店或者网页搜索这些工具，这些信息也会作为自动创建的输入，生成的模板会带上工具的信息，例如让模型
在xxx情况下调用什么工具，以及调用工具前需要用户提供哪些参数信息，能够一定程度上提升工具调用的准确率（在完全不进行修改的情况下，提升10%，一般用户会根据自动生成的内容进行修改，自动生成的内容作为骨架和格式引导，引导用户根据实际需求去修改prompt）。
还有一种算法是交互式创建的提示词，通过一些引导式的提问，根据用户的回复去不断完善整个prompt。比如，首先会问用户希望构建一个什么样的智能助手，用户会回复想要创建一个作业辅导助手，然后会根据用户场景去追问详细的内容，比如需要什么学科的助手，
以及小学或者初中，然后询问是否存在一些约束限制条件，以及偏好用什么样的语气回复。通过不断引导用户提出个性化的需求，完成整个提示词的创建。上面讲的是实现的最终效果，那在背后的实现算法，我们有根据不同模型的能力强弱，实现两种不同的类型，
一种是全局指令类型，多agent协同类型。全局指令类型适合上下文理解能力和遵循能力比较好的模型，所谓全局指令，就是说我用一个system prompt指令给到大模型，告诉他怎么去和用户进行交互，第一个步骤问什么，第二个步骤怎么反问，以及怎么根据用户的回复去更新
生成的提示词。大模型通过理解这个system prompt指令，就能够完成我们设定的对话过程，和用户去交互，完成prompt构建；第二种是多agent协同，这种类型适合能力稍弱的大模型，这种大模型就是一次让它做单个任务的效果还可以，但是如果你有一堆复杂的指令，告诉他很多
执行步骤，他可能就混乱了，无法按照给定的指令来和用户交互。所以我们把它做成一个多agent的架构来和用户对话，完成交互式的创建流程。我们主要用了一个主对话的agent加上若干个执行特定任务的agent，主对话的agent工作很简单，就是让它按照既定的交互步骤用户进行对话，
但是它不做任何具体的提示词生成的操作，当它收到用户回复之后，会调用很多别的agent来生成提示词的某些部分，例如生成角色描述和生成技能、约束条件这些内容的时候，都是调用的一个独立的agent来完成，它也可以调用一个修改提示词的agent，根据用户反馈意见，对提示词
进行修改优化。上面说的是提示词自动创建这一块的算法。
除此之外，我还做了一些提示词自动优化方面的工作，这个主要是借鉴到一些论文中提出的方法，再结合我们的业务来做的。这个功能主要是允许用户在实际使用agent的过程中，发现的一些bad case来进行优化。用户识别到bad case之后，可以对模型输出进行纠正，
填写他们期望的模型输出，然后把这些case作为这个自优化算法的输入，最终这个算法会在这些case上进行迭代优化，返回一个效果更好的提示词。具体的算法实现上，我们主要分成两类，一类是文本梯度，一类是遗传变异。文本梯度算法的主要思想是利用大模型自身去对bad case进行
分析，并通过反思，分析当前提示词中存在的问题，并基于这些问题分析，去修改prompt。所以这个算法有几个步骤，首先是前向推理，让大模型根据case的输入推理出回复，第二步是loss评估，把第一个步骤推理出的回复以及用户填写的期望回复作为输入，让大模型分析哪些case存在
问题，第三个步骤是文本梯度生成，结合上个步骤中分析出的bad case，让大模型进一步去分析，原始的提示词中有哪些内容导致了bad case的产生；第四个步骤是提示词修改，根据上个步骤的分析结果，来修改提示词。除了