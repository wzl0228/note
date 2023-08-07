from transformers import AutoTokenizer, AutoModel, AutoConfig
import torch
import os
import sys
import jieba
import json
import argparse
import rouge

sys.setrecursionlimit(5000) # 解决rouge计算的递归限制问题，python默认递归限制是1000
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
config = AutoConfig.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True, pre_seq_len=256)
model = AutoModel.from_pretrained("THUDM/chatglm-6b", config=config, trust_remote_code=True).half().cuda()
prefix_state_dict = torch.load(os.path.join("ptuning/output/adgen-chatglm-6b-pt-Sugg-256-2e-2/checkpoint-600", "pytorch_model.bin"))
new_prefix_state_dict = {}
for k, v in prefix_state_dict.items():
    if k.startswith("transformer.prefix_encoder."):
        new_prefix_state_dict[k[len("transformer.prefix_encoder."):]] = v
model.transformer.prefix_encoder.load_state_dict(new_prefix_state_dict)

# chatglm
def chatglm(inputs):
    summary, history = model.chat(tokenizer, inputs, history=[])
    # print(response)
    # assert 0
    return summary

def compute_rouge(source, target):
    """计算rouge-1、rouge-2、rouge-l
    """
    # source, target = ' '.join(source), ' '.join(target)
    source, target = ' '.join(jieba.cut(source, cut_all=False)), ' '.join(jieba.cut(target, cut_all=False)) # 换成jieba分词计算rouge，否则会递归深度超限
    try:
        scores = rouge.Rouge().get_scores(hyps=source, refs=target)
        return {
            'rouge-1': scores[0]['rouge-1']['f'],
            'rouge-2': scores[0]['rouge-2']['f'],
            'rouge-l': scores[0]['rouge-l']['f'],
        }
    except ValueError:
        return {
            'rouge-1': 0.0,
            'rouge-2': 0.0,
            'rouge-l': 0.0,
        }
    
    
def compute_rouges(sources, targets):
    scores = {
        'rouge-1': 0.0,
        'rouge-2': 0.0,
        'rouge-l': 0.0,
    }
    for source, target in zip(sources, targets):
        score = compute_rouge(source, target)
        for k, v in scores.items():
            scores[k] = v + score[k]

    return {k: v / len(targets) for k, v in scores.items()}


# 意见->整改意见
def SugS(args):
    data_json = [] # 存储结果，最终写入结果json文件
    gens, summaries = [], []
    example = json.load(open(args.data_path))

    for feature in example:
        raw_data = feature['text']
        gen = chatglm(raw_data)

        # 如果生成的gen和summary的内容rouge值计算小于某个阈值，则重新生成一次，可以调整重新生成的次数
        re_gen = 0 # 记录该条数据是否重新生成
        source, target = ' '.join(jieba.cut(gen, cut_all=False)), ' '.join(jieba.cut(feature['summary'], cut_all=False))
        rouges = rouge.Rouge().get_scores(hyps=source, refs=target)
        rouge_1 = rouges[0]["rouge-1"]['f'] * 100
        rouge_2 = rouges[0]["rouge-2"]['f'] * 100
        rouge_l = rouges[0]["rouge-l"]['f'] * 100
        if rouge_1 < 30 or rouge_l < 30:
            re_gen = 1
            gen = chatglm(raw_data)
        
        gens.append(gen)
        item = {'text': raw_data,
                'summary': feature['summary'],
                'generate': gen,
                'rouge-1': rouge_1,
                'rouge-2': rouge_2,
                'rouge-l': rouge_l,
                're-gen': re_gen}
        data_json.append(item)
        if 'summary' in feature:
            summaries.append(feature['summary'])
    
    with open(args.result_file, 'w', encoding='utf-8') as json_file:
        json.dump(data_json, json_file, ensure_ascii=False, indent=2)

    if summaries:
        scores = compute_rouges(gens, summaries)
        print(scores)
    print('完成!')

if __name__ == '__main__':    
    parser = argparse.ArgumentParser()
    parser.add_argument("-data_path", default='ptuning/data0805/Sugg/test.json', type=str)
    parser.add_argument('--result_file', default='ptuning/data0805/Sugg/test_result.json')
    args = parser.parse_args()
    SugS(args)
