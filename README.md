完全云端运行：使用谷歌CoLaboratory训练神经网络
Colaboratory 是一个 Google 研究项目，旨在帮助传播机器学习培训和研究成果。它是一个 Jupyter 笔记本环境，不需要进行任何设置就可以使用，并且完全在云端运行。Colaboratory 笔记本存储在 Google 云端硬盘 (https://drive.google.com/) 中，并且可以共享，就如同您使用 Google 文档或表格一样。Colaboratory 可免费使用。本文介绍如何使用 Google CoLaboratory 训练神经网络。
工具链接：https://colab.research.google.com/

谷歌近期上线了协作写代码的内部工具 Google CoLaboratory。Colaboratory 是一个 Google 研究项目，旨在帮助传播机器学习培训和研究成果。它是一个 Jupyter 笔记本环境，不需要进行任何设置就可以使用，并且完全在云端运行。

Colaboratory 笔记本存储在 Google 云端硬盘 (https://drive.google.com/) 中，并且可以共享，就如同您使用 Google 文档或表格一样。Colaboratory 可免费使用。

CoLaboratory
首先，访问 CoLaboratory 网站（http://g.co/colab），注册后接受使用该工具的邀请。确认邮件通常需要一天时间才能返回你的邮箱。CoLaboratory 允许使用谷歌虚拟机执行机器学习任务和构建模型，无需担心计算力的问题，而且它是免费的。

打开 CoLaboratory，会出现一个「Hello, Colaboratory」文件，包含一些基本示例。建议尝试一下。

使用 CoLaboratory 可以在 Jupyter Notebook 上写代码。写好后执行 (Shift + Enter)，代码单元下方就会生成输出。



除了写代码，CoLaboratory 还有一些技巧（trick）。你可以在 notebook 中 shell 命令前加上「!」。如：!pip install -q keras。这样你就可以很大程度上控制正在使用的谷歌虚拟机。点击左上方（菜单栏下）的黑色按钮就可以找到它们的代码片段。

本文旨在展示如何使用 CoLaboratory 训练神经网络。我们将展示一个在威斯康星乳腺癌数据集上训练神经网络的示例，数据集可在 UCI Machine Learning Repository（http://archive.ics.uci.edu/ml/datasets）获取。本文的示例相对比较简单。

本文所用的 CoLaboratory notebook 链接：https://colab.research.google.com/notebook#fileId=1aQGl_sH4TVehK8PDBRspwI4pD16xIR0r

深度学习
深度学习是一种机器学习技术，它使用的计算技术一定程度上模仿了生物神经元的运行。各层中的神经元网络不断将信息从输入传输到输出，直到其权重调整到可以生成反映特征和目标之间底层关系的算法。

要想更多地了解神经网络，推荐阅读这篇论文《Artificial Neural Networks for Beginners》（https://arxiv.org/pdf/cs/0308031.pdf）。

代码
问题：研究者获取乳房肿块的细针穿刺（FNA），然后生成数字图像。该数据集包含描述图像中细胞核特征的实例。每个实例包括诊断结果：M（恶性）或 B（良性）。我们的任务是在该数据上训练神经网络根据上述特征诊断乳腺癌。
