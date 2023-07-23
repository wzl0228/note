# Generating EDU Extracts for Plan-Guided Summary Re-Ranking

## 论文摘要
二步方法，是指在生成摘要后重新排序，以返回单个摘要的方法，与比标准一步方法相比可以提高ROUGE得分。然而，标准解码方法(如波束搜索（beam search）、核心样本（nucleus）、多样化波束搜索（diverse beam search）)会产生冗余，往往是质量较低的内容。为了解决该问题，我们设计了一种新方法通过重新排序生成摘要候选者。我们将每个候选摘要建立在其独特的内容计划上，并使用模型的top beam生成计划指导(plan-guided)的摘要。具体来说，一个标准语言模型(一个BART LM)自回归地使用提取复制机制生成基本语言单元(Elemental Discourse Unit，EDU)内容计划。内容计划生成器中的top K beams然后将用于指导一个单独的LM，为每个特定计划生成一个单独抽象候选者。我们将重新排序器(BRIO)应用于我们生成摘要候选者的过程，也是我们解码方法的基准。我们的方法在CNN / Dailymail、NYT和Xsum中，ROUGE-2 F1得分分别提高了0.88、2.01和0.38。CNN/DM的人类评估验证了这些结果。类似地，在CNN/DM中的1,000个样本中，我们发现提示GPT-3按照EDU计划比基于样本的方法在ROUGE-2 F1得分上提高了1.05个点。

## 
