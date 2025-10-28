import os
import re
import json
import glob
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from pypinyin import lazy_pinyin
from transformers import AutoTokenizer
from bs4 import BeautifulSoup
from transformers import AutoTokenizer, BertModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import AgglomerativeClustering
from tools import normalized_levenshtein_distance, jaccard_similarity, longest_common_substring, iterative_lcs_similarity


def read_file(path):
    with open(path, "r",encoding='utf-8') as f:
        lines = f.readlines()
    return "".join(lines)


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = json.load(f)
    return content


def write_json(path, context):
    with open(path,'w') as f:
        json.dump(context,f)


def get_name_list(output, args):
    name_score_dic = {}
    output = output.replace('[', '\n').replace(']', '\n').replace('{', '\n').replace('}', '\n').replace(',', '\n').replace('，', '\n').replace('、', '\n').replace('。', '\n').replace('“', '\n').replace('”', '\n').replace('"', '\n').split()
    # print(output)
    if args.lang == 'en':
        pa = r'^[0-9]+$'
    else:
        pa = r'^[0-9A-Za-z]+$'
    for item in output:
        if '抱歉' not in item and '无法' not in item and '未找到' not in item and '没有' not in item and item != '无':
            item = item.split(":")[-1].split("：")[-1].split("-")[0]
            if len(re.findall(pa, item))==0:
                if item in name_score_dic.keys():
                    name_score_dic[item] += 1
                else:
                    name_score_dic[item] = 1
    return name_score_dic


def jude_name_true(pname, tname, args):
    pword = pname.split(' ')[0].lower()
    tword = tname.split(' ')[0].lower()
    pname = pname.lower().replace(' ','')
    tname = tname.lower().replace(' ','')
    full_pinyin = ''.join(lazy_pinyin(tname)).replace('（','(').replace('）',')').replace('“','"').replace('”','"')
    if len(pname)>0 and len(tname)>0 and iterative_lcs_similarity(pname, tname)==1:
        return 1
    elif len(pname)>0 and len(full_pinyin)>0 and iterative_lcs_similarity(pname, full_pinyin)==1:
        return 1
    elif len(pword)>0 and len(tword)>0 and pword == tword:
        return 1
    else:
        # print(lanme,tname)
        return 0


def main(args):
    print(args.res_path)
    
    TP = 0
    FP = 0
    FN = 0
    for item in tqdm(os.listdir(args.res_path)):
        # print(f'------{item}------')
        name = item.split('_')[0]
        label = item.split('.html')[0].replace('\\','/').split('_')[1:]
        
        if args.mode == 'vanilla':
            data = read_file(os.path.join(args.res_path, item))
            name_score_dic = get_name_list(data, args)
            sorted_list = sorted(name_score_dic.items(), key=lambda x: x[1], reverse = True)
        sorted_list = read_json(os.path.join(args.res_path, item))
        # print(f'disambiguated and sorted output: {sorted_list}')
        
        flag = 0
        if len(sorted_list) > 0:
            score_max_list = []
            for t in sorted_list:
                if t[1] == sorted_list[0][1]:
                    score_max_list.append(t[0])
            str_len = 0
            for n in score_max_list:
                if str_len < len(n):
                    result = n
                    str_len = len(n)
            for l in label:
                if jude_name_true(result, l, args):
                    flag += 1
            if flag > 0:
                T += 1
            else:
                FP += 1
        else:
            FN += 1
    
    pre = TP/(TP+FP)
    rec = TP/(TP+FN)
    f1 = 2 * pre * rec / (pre + rec)
    print(f'TP: {TP}  FP: {FP}  FN: {FN}')
    print(f'pre: {pre}  rec: {rec}  f1: {f1}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--res_path', type=str, default='./data/woi_cn/OwnerHunter/ranking')
    parser.add_argument('--mode', type=str, default='OwnerHunter')
    args = parser.parse_args()
    main(args)