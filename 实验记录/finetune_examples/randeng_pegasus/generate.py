from transformers import PegasusForConditionalGeneration, BertTokenizer
from tokenizers_pegasus import PegasusTokenizer
import jieba
from transformers import BertTokenizer, BatchEncoding
from torch._six import container_abcs, string_classes, int_classes
import torch
from torch.utils.data import DataLoader, Dataset
import re
import os
import csv
import json
import argparse
from tqdm.auto import tqdm
from multiprocessing import Pool, Process
import pandas as pd
import numpy as np
import rouge

os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def load_data(data_path):
    """加载数据
    单条格式：(summary, text) # (title, content)->(summary, text)
    """
    D = []
    example = json.load(open(data_path))
    for i in example:
        text = i['text']
        summary = i['summary']
        D.append((summary, text))
    return D
    

# class T5PegasusTokenizer(BertTokenizer):
#     """结合中文特点完善的Tokenizer
#     基于词颗粒度的分词，如词表中未出现，再调用BERT原生Tokenizer
#     """
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
        
#     def pre_tokenizer(self, x):
#         return jieba.cut(x, HMM=False)

#     def _tokenize(self, text, *arg, **kwargs):
#         split_tokens = []
#         for text in self.pre_tokenizer(text):
#             if text in self.vocab:
#                 split_tokens.append(text)
#             else:
#                 split_tokens.extend(super()._tokenize(text))
#         return split_tokens
    

class KeyDataset(Dataset):
    def __init__(self, dict_data):
        self.data = dict_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data[index]
    
    
def create_data(data, tokenizer, max_len):
    """调用tokenizer.encode编码正文/标题，每条样本用dict表示数据域
    """
    ret, flag, summary = [], True, None
    for text in data:
        if type(text) == tuple:
            summary, text = text
        text_ids = tokenizer.encode(text, max_length=max_len,
                                    truncation='only_first')

        if flag:
            flag = False
            print(text)

        features = {'input_ids': text_ids,
                    'attention_mask': [1] * len(text_ids),
                    'raw_data': text}
        if summary:
            features['summary'] = summary
        ret.append(features)
    return ret


def sequence_padding(inputs, length=None, padding=0):
    """Numpy函数，将序列padding到同一长度
    """
    if length is None:
        length = max([len(x) for x in inputs])

    pad_width = [(0, 0) for _ in np.shape(inputs[0])]
    outputs = []
    for x in inputs:
        x = x[:length]
        pad_width[0] = (0, length - len(x))
        x = np.pad(x, pad_width, 'constant', constant_values=padding)
        outputs.append(x)

    return np.array(outputs, dtype='int64')


def default_collate(batch):
    """组batch
    各个数据域分别转换为tensor，tensor第一个维度等于batch_size
    """
    np_str_obj_array_pattern = re.compile(r'[SaUO]')
    default_collate_err_msg_format = (
        "default_collate: batch must contain tensors, numpy arrays, numbers, "
        "dicts or lists; found {}")
    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out).to(device)
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int_classes):
        return torch.tensor(batch, dtype=torch.long)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, container_abcs.Mapping):
        return {key: default_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, container_abcs.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            batch = sequence_padding(batch)

        return default_collate([default_collate(elem) for elem in batch])

    raise TypeError(default_collate_err_msg_format.format(elem_type))
    

def prepare_data(args, tokenizer):
    """准备batch数据
    """
    test_data = load_data(args.test_data)
    test_data = create_data(test_data, tokenizer, args.max_len)
    test_data = KeyDataset(test_data)
    test_data = DataLoader(test_data, batch_size=args.batch_size, collate_fn=default_collate)
    return test_data


def compute_rouge(source, target):
    """计算rouge-1、rouge-2、rouge-l
    """
    
    source, target = ' '.join(source), ' '.join(target)
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


def generate(test_data, model, tokenizer, args):
    data_json = [] # 存储结果，最终写入结果json文件
    gens, summaries = [], []
    
    # with open(args.result_file, 'w', encoding='utf-8', newline='') as f:
    #     writer = csv.writer(f, delimiter='\t')
    #     model.eval()
    #     for feature in tqdm(test_data):
    #         raw_data = feature['raw_data']
    #         content = {k : v for k, v in feature.items() if k not in ['raw_data', 'summary']} 
    #         gen = model.generate(max_length=args.max_len_generate,
    #                             eos_token_id=tokenizer.sep_token_id,
    #                             decoder_start_token_id=tokenizer.cls_token_id,
    #                             **content)
    #         gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
    #         gen = [item.replace(' ', '') for item in gen]
    #         writer.writerows(zip(gen, raw_data))
    #         gens.extend(gen)
    #         if 'summary' in feature:
    #             summaries.extend(feature['summary'])

    model.eval()
    for feature in tqdm(test_data):
        # print(feature)
        # assert 0
        raw_data = feature['raw_data']
        content = {k : v for k, v in feature.items() if k not in ['raw_data', 'summary']} 
        gen = model.generate(max_length=args.max_len_generate,
                            eos_token_id=tokenizer.sep_token_id,
                            decoder_start_token_id=tokenizer.cls_token_id,
                            **content)
        gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
        gen = [item.replace(' ', '') for item in gen]
        # writer.writerows(zip(gen, raw_data))
        gens.extend(gen)
        for idx in range(len(raw_data)):
            item = {'text': raw_data[idx],
                    'summary': feature['summary'][idx],
                    'generate': gen[idx]}
            data_json.append(item)
        if 'summary' in feature:
            summaries.extend(feature['summary'])
    
    with open(args.result_file, 'w', encoding='utf-8') as json_file:
        json.dump(data_json, json_file, ensure_ascii=False, indent=2)

    if summaries:
        scores = compute_rouges(gens, summaries)
        print(scores)
    print('完成!')


def generate_multiprocess(feature):
    """多进程
    """
    model.eval()
    raw_data = feature['raw_data']
    content = {k: v for k, v in feature.items() if k != 'raw_data'}
    gen = model.generate(max_length=args.max_len_generate,
                             eos_token_id=tokenizer.sep_token_id,
                             decoder_start_token_id=tokenizer.cls_token_id,
                             **content)
    gen = tokenizer.batch_decode(gen, skip_special_tokens=True)
    results = ["{}\t{}".format(x.replace(' ', ''), y) for x, y in zip(gen, raw_data)]
    return results


def init_argument():
    parser = argparse.ArgumentParser(description='Randeng-Pegasus-523M-Summary-Chinese')
    parser.add_argument('--test_data', default='filtered_data/test.json')
    # parser.add_argument('--result_file', default='filtered_data/test_result.tsv')
    parser.add_argument('--result_file', default='filtered_data/test_result.json')
    parser.add_argument('--pretrain_model', default='Randeng-Pegasus-523M-Summary-Chinese')    
    parser.add_argument('--model', default='saved_model/summary_model')

    parser.add_argument('--batch_size', default=1, help='batch size') # 16->1
    parser.add_argument('--max_len', default=256, help='max length of inputs')
    parser.add_argument('--max_len_generate', default=256, help='max length of generated text')
    parser.add_argument('--use_multiprocess', default=False, action='store_true')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    
    # step 1. init argument
    args = init_argument()

    # step 2. prepare test data
    tokenizer = PegasusTokenizer.from_pretrained(args.pretrain_model)
    test_data = prepare_data(args, tokenizer)
    
    # step 3. load finetuned model
    model = torch.load(args.model, map_location=device)

    # step 4. predict
    res = []
    if args.use_multiprocess and device == 'cpu':
        print('Parent process %s.' % os.getpid())
        p = Pool(2)
        res = p.map_async(generate_multiprocess, test_data, chunksize=2).get()
        print('Waiting for all subprocesses done...')
        p.close()
        p.join()
        res = pd.DataFrame([item for batch in res for item in batch])
        res.to_csv(args.result_file, index=False, header=False, encoding='utf-8')
        print('Done!')
    else:
        generate(test_data, model, tokenizer, args)

# 抽取最长长度为256的数据，对应的rouge指标
# {'rouge-1': 0.37074567708847656, 'rouge-2': 0.20188447384849184, 'rouge-l': 0.31370841377729625}