CoLaboratory
首先，访问 CoLaboratory 网站（http://g.co/colab），注册后接受使用该工具的邀请。确认邮件通常需要一天时间才能返回你的邮箱。CoLaboratory 允许使用谷歌虚拟机执行机器学习任务和构建模型，无需担心计算力的问题，而且它是免费的。

打开 CoLaboratory，会出现一个「Hello, Colaboratory」文件，包含一些基本示例。建议尝试一下。

使用 CoLaboratory 可以在 Jupyter Notebook 上写代码。写好后执行 (Shift + Enter)，代码单元下方就会生成输出。
[img](https://image.jiqizhixin.com/uploads/wangeditor/3ccfa35d-df58-4cd1-97bd-9756df94e235/731851.png)
除了写代码，CoLaboratory 还有一些技巧（trick）。你可以在 notebook 中 shell 命令前加上「!」。如：!pip install -q keras。这样你就可以很大程度上控制正在使用的谷歌虚拟机。点击左上方（菜单栏下）的黑色按钮就可以找到它们的代码片段。

本文旨在展示如何使用 CoLaboratory 训练神经网络。我们将展示一个在威斯康星乳腺癌数据集上训练神经网络的示例，数据集可在 UCI Machine Learning Repository（http://archive.ics.uci.edu/ml/datasets）获取。本文的示例相对比较简单。

本文所用的 CoLaboratory notebook 链接：https://colab.research.google.com/notebook#fileId=1aQGl_sH4TVehK8PDBRspwI4pD16xIR0r


代码
问题：研究者获取乳房肿块的细针穿刺（FNA），然后生成数字图像。该数据集包含描述图像中细胞核特征的实例。每个实例包括诊断结果：M（恶性）或 B（良性）。我们的任务是在该数据上训练神经网络根据上述特征诊断乳腺癌。

打开 CoLaboratory，出现一个新的 untitled.ipynb 文件供你使用。

谷歌允许使用其服务器上的一台 linux 虚拟机，这样你可以访问终端为项目安装特定包。如果你只在代码单元中输入 !ls 命令（记得命令前加!），那么你的虚拟机中会出现一个 datalab 文件夹。



我们的任务是将数据集放置到该机器上，这样我们的 notebook 就可以访问它。

Keras
Keras 是一种构建人工神经网络的高级 API。它使用 TensorFlow 或 Theano 后端执行内部运行。要安装 Keras，必须首先安装 TensorFlow。CoLaboratory 已经在虚拟机上安装了 TensorFlow。使用以下命令可以检查是否安装 TensorFlow：

!pip show tensorflow

你还可以使用!pip install tensorflow==1.2，安装特定版本的 TensorFlow。

另外，如果你更喜欢用 Theano 后端，可以阅读该文档：https://keras.io/backend/。

安装 Keras：

!pip install -q keras

# Importing the Keras libraries and packages

import keras

from keras.models import Sequential

from keras.layers import Dense

使用 Sequential 和 Dense 类别指定神经网络的节点、连接和规格。如上所示，我们将使用这些自定义网络的参数并进行调整。

为了初始化神经网络，我们将创建一个 Sequential 类的对象。

# Initialising the ANN

classifier = Sequential()

现在，我们要来设计网络。

对于每个隐藏层，我们需要定义三个基本参数：units、kernel_initializer 和 activation。units 参数定义每层包含的神经元数量。Kernel_initializer 定义神经元在输入数据上运行时的初始权重（详见 https://faroit.github.io/keras-docs/1.2.2/initializations/）。activation 定义数据的激活函数。

注意：如果现在这些项非常大也没事，很快就会变得更加清晰。

第一层：

16 个具备统一初始权重的神经元，激活函数为 ReLU。此外，定义参数 input_dim = 30 作为输入层的规格。注意我们的数据集中有 30 个特征列。

Cheat：

我们如何确定这一层的单元数？人们往往会说这需要经验和专业知识。对于初学者来说，一种简单方式是：x 和 y 的总和除以 2。如 (30+1)/2 = 15.5 ~ 16，因此，units = 16。

第二层：第二层和第一层一样，不过第二层没有 input_dim 参数。

输出层：由于我们的输出是 0 或 1，因此我们可以使用具备统一初始权重的单个单元。但是，这里我们使用 sigmoid 激活函数。

# Adding the input layer and the first hidden layer

classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu', input_dim = 30))



# Adding the second hidden layer

classifier.add(Dense(units = 16, kernel_initializer = 'uniform', activation = 'relu'))



# Adding the output layer

classifier.add(Dense(units = 1, kernel_initializer = 'uniform', activation = 'sigmoid'))



# Compiling the ANN

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

拟合：

运行人工神经网络，发生反向传播。你将在 CoLaboratory 上看到所有处理过程，而不是在自己的电脑上。

# Fitting the ANN to the Training set

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)

这里 batch_size 是你希望同时处理的输入量。epoch 指数据通过神经网络一次的整个周期。它们在 Colaboratory Notebook 中显示如下：

![img](https://image.jiqizhixin.com/uploads/wangeditor/3ccfa35d-df58-4cd1-97bd-9756df94e235/473055.png)

进行预测，构建混淆矩阵。

# Predicting the Test set results

y_pred = classifier.predict(X_test)

y_pred = (y_pred > 0.5)



# Making the Confusion Matrix

from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_test, y_pred) 

训练网络后，就可以在 X_test set 上进行预测，以检查模型在新数据上的性能。在代码单元中输入和执行 cm 查看结果。

混淆矩阵
混淆矩阵是模型做出的正确、错误预测的矩阵表征。该矩阵可供个人调查哪些预测和另一种预测混淆。这是一个 2×2 的混淆矩阵。

![img](https://image.jiqizhixin.com/uploads/wangeditor/3ccfa35d-df58-4cd1-97bd-9756df94e235/595146.png)


混淆矩阵如下所示。

![img](https://image.jiqizhixin.com/uploads/wangeditor/3ccfa35d-df58-4cd1-97bd-9756df94e235/687177.png)

上图表示：70 个真负类、1 个假正类、1 个假负类、42 个真正类。很简单。该平方矩阵的大小随着分类类别的增加而增加。
