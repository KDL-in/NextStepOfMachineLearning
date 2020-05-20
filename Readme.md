# Readme

这是李宏毅老师ML lectures 2020的课程作业的实现。真心感谢老师和他们团队的助教们。本项目只能作为思路参考。

基本是读懂示例代码做了实现，这里也不提供strong baseline的突破方法。因为作业量比较大，全程用Jupyter Notebook写的，方便梳理逻辑结构，所以执行非线性，代码应该都不能直接跑的（反作弊）。

另外一点就是，前期不熟悉pytorch做的实现比较糟糕，后期慢慢学到了很多东西。

> - HW1，线性回归，纯numpy实现
>
> - HW2，概率分类模型及逻辑回归
>
>   - Probabilistic Generative Model
>   - Logistic Regression
>
> - HW3， CNN食物分类
>
>   - CNN深度、参数量的探究
>   - 混淆矩阵，confusion     matrix 的使用以及分析
>
> - HW4，Semi-supervised Learning
>
>   - Word2vec的训练和使用
>   - Pytorch LSTM的使用
>   - Bow，bag of   word模型实现
>   - Self-learning 实现
>   - 自定义损失函数
>
> - HW5， 神经网络的解释
>
>   - Saliency map，loss对x的梯度
>   - Filter  activation，从torch model中截断某层的输出
>   - Filter  visualization，filter可视化，训练得到特定x，最大化filter输出，可视化filter
>   - Lime，线性模型解释复杂模型的局部
>
> - HW6，神经网络攻击
>
>   - FGSM攻击
>   - 黑箱攻击，攻击在线模型
>
> - HW7，网络压缩
>
>   - Architecture Design
>      - depthwise cnn
>     - pointwise cnn
>     - Knowledge Dsitillation
>        - kl散度
>       - 自定义loss
>        - train方法，微调，run_epoch
>     - Network Pruning
>       - 裁剪网络参数
>       - 重要性衡量
>  - Weight Quantization
>       - 网络数据类型转换
>  
> - HW8 ，机器翻译任务，seq2seq，这个任务的实现难度比较大
>   - seq2seq基本框架
>    - attention实现
>     - Schedule Sampling 实现
>    - Beam Search实现，未验证
>     - Bleu评价

