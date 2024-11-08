UCAS大数据分析编程作业:完成一种基于深度学习的命名实体识别方法
- 主体模型使用bert
- results/ 下包括所有模型的收敛曲线
- 包括bert, bert-crf, bert-cnn, bert-lstm, bert-cnn-lstm模型

**开发环境**:
- Python 3.11.10
- Ubuntu 18.04.1
- cuda 12.1.105
- torch2.4.0+cu121
- NVIDIA GeForce RTX 3090

**结果展示:**
| model&trick                 | accuracy | $f_1$ score |
| --------------------------- | -------- | ----------- |
| bert+softmax                | 0.9638   | 0.8223      |
| bert+softmax+余弦退火学习率 | 0.9668   | 0.8456      |
| bert+lstm                   | 0.9594   | 0.8009      |
| bert+cnn                    | 0.9648   | 0.8128      |
| bert+lstm+cnn               | 0.9778   | 0.8606      |

![bert_fuse](https://gitee.com/meetzyj/typora-images/raw/master/imgs/20241107203411.png)

**声明:**

本项目基于https://github.com/xsfmGenius/Ner_Bert_CoNLL-2003 进行开发
