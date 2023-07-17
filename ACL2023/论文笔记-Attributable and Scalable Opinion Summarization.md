# Attributable and Scalable Opinion Summarization

## 评论聚合（意见摘要）面临的挑战
1. 很难获得参考摘要，因此模型训练总是再缺乏gold standard references基础上训练的（21年以前）
2. 热门产品（entities）可能有数百个评论，如果方法的可扩展性差，可能会造成计算困难
3. 我们期望得到的意见摘要应该是抽象的，剔除一些不必要的细节，但也不能产生包含虚假信息“幻觉”。理想情况下，模型应该是可归因的，提供一些证据来证明其输出（生成摘要）的合理性。

> 先前的工作一部分可归因可扩展，但是生成的摘要太具体（我们期望是剔除unnecessarily specific details），另一部分生成摘要太抽象包含幻觉，且缺乏扩展性。

## 本论文思想
生成抽象摘要的同时，附上对输入句子的引用，这些引用作为每个输出句子的证据验证输入评论的哪些部分被用于生成摘要。

具体为：
1. 将评论中自然语言句子编码为（在分层离散潜在空间的）路径。
2. 给定关于某个特定实体的多条评论语句，确定在这些评论之间共享的公共子路径（common subpaths），并将这些子路径解码回自然语言，生成摘要。这些包含（选出的）子路径（编码后）的句子，充当生成句子的证据。

![image](https://github.com/wzl0228/note/assets/80249917/69bf83f8-c3b0-4b14-959e-91c95db89b84)

训练（上）：将评论中的句子编码为（在分层离散潜在空间的）路径。如将“Tasty burgers” 编码为（中间部分）像“树”一样形状的其中一部分，在图中用黑色实线表示。

推理（下）：对输入评论中的所有句子进行编码，并确定用于摘要的频繁路径或子路径（底部）。来自三个示例输入的一致意见是，食物是好的，因此红色显示的子路径是重复的；
解码它应该会产生类似“Good food”的输出。

HERCULES is trained to encode sentences
from reviews as paths through a hierarchical discrete
latent space (top). At inference time, we encode all
sentences from the input reviews, and identify frequent
paths or subpaths to use for the summary (bottom). The
consensus opinion from the three example inputs is that
the food is good, so the subpath shown in red is repeated;
decoding it should result in an output like "Good food".

