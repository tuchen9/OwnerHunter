import os
import re
import json
import glob
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
from bs4 import BeautifulSoup
from pypinyin import lazy_pinyin
from transformers import AutoTokenizer, BertModel
from sklearn.cluster import AgglomerativeClustering
from tools import longest_common_substring, iterative_lcs_similarity


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


def get_name_score_list(output):
    name_score_dic = {}
    output = output.replace('[', '\n').replace(']', '\n').replace('{', '\n').replace('}', '\n').replace(',', '\n').replace('，', '\n').replace('、', '\n').replace('。', '\n').replace('“', '\n').replace('”', '\n').replace('"', '\n').split()
    # print(output)
    pa = r'^[0-9A-Za-z]+$'
    for item in output:
        if '抱歉' not in item and '无法' not in item and '未找到' not in item and '没有' not in item and item != '无':
            tmp = item.split(":")[-1].split("：")[-1].split("-")
            if len(tmp)>1:
                try:
                    if len(tmp[0])>1 and len(re.findall(pa, tmp[0]))==0:
                        if tmp[0] in name_score_dic.keys():
                            name_score_dic[tmp[0]] += float(tmp[1])
                        else:
                            name_score_dic[tmp[0]] = float(tmp[1])
                except ValueError:
                    # print('no score')
                    pass
    return name_score_dic


def get_name_score_list_en(output):
    name_score_dic = {}
    pattern1 = r'{(.*?)}'
    pattern2 = r'[^\d{}]+-0\.\d+'
    pattern3 = r'[^\d{}]+-1'
    pattern4 = r'^[a-zA-Z .,&-]+$'
    output=output.replace('[', '{').replace(']', '}').replace('(', '{').replace(')', '}').replace('【', '{').replace('】', '}').replace('（', '{').replace('）', '}').replace('“', '{').replace('”', '}').replace('。', '\n').split('\n')
    for o in output:
        matches = re.findall(pattern1, o)
        # print(matches)
        if len(matches)==0:
            o = o.split(":")[-1].split("：")[-1]
            if '抱歉' not in o and '无法' not in o and o != '无':
                matches_ns = re.findall(pattern2, o) + re.findall(pattern3, o)
                for ns in matches_ns:
                    tmp = ns.split("-")
                    tmp_name = '-'.join(tmp[:-1])
                    if len(tmp_name)>1:
                        if tmp_name in name_score_dic.keys():
                            name_score_dic[tmp_name] += float(tmp[-1])
                        else:
                            name_score_dic[tmp_name] = float(tmp[-1])
        for item in matches:
            item = item.split(":")[-1].split("：")[-1]
            if '抱歉' not in item and '无法' not in item and item != '无':
                matches_ns = re.findall(pattern2, item) + re.findall(pattern3, item)
                for ns in matches_ns:
                    tmp = ns.split("-")
                    tmp_name = '-'.join(tmp[:-1])
                    if len(tmp_name)>1:
                        if tmp_name in name_score_dic.keys():
                            name_score_dic[tmp_name] += float(tmp[-1])
                        else:
                            name_score_dic[tmp_name] = float(tmp[-1])
    return name_score_dic


def disambiguation(mentions):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    n = len(mentions)
    cs_sim_matrix = torch.zeros((n, n)).to(device)
    for i in range(n):
        for j in range(i + 1, n):
            cs_sim_matrix[i, j] = cs_sim_matrix[j, i] = iterative_lcs_similarity(mentions[i], mentions[j])
    
    bert_path = './RoBERTa_wwm_ext/'
    bert_tokenizer = AutoTokenizer.from_pretrained(bert_path)
    bert_model = BertModel.from_pretrained(bert_path).to(device)
    with torch.no_grad():
        batch = bert_tokenizer(mentions, max_length=512, truncation=True, padding='max_length', return_tensors="pt").to(device)
        embeddings = bert_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], token_type_ids=batch['token_type_ids'])[1]
    embeddings = F.normalize(embeddings, dim=-1)
    cos_sim = torch.mm(embeddings, embeddings.permute(1,0))
    # print('----')
    # print(mentions)
    # HAC_model = AgglomerativeClustering(n_clusters=None, distance_threshold=0.3, metric='precomputed', linkage='average')
    # labels = HAC_model.fit_predict(1-(1*cs_sim_matrix+0*cos_sim).cpu())
    # print(f'1:0 {labels}')
    HAC_model = AgglomerativeClustering(n_clusters=None, distance_threshold=0.3, metric='precomputed', linkage='average')
    labels = HAC_model.fit_predict(1-(0.9*cs_sim_matrix+0.1*cos_sim).cpu())
    # print(f'0.9:0.1 {labels}')
    # HAC_model = AgglomerativeClustering(n_clusters=None, distance_threshold=0.3, metric='precomputed', linkage='average')
    # labels = HAC_model.fit_predict(1-(0.8*cs_sim_matrix+0.2*cos_sim).cpu())
    # print(f'0.8:0.2 {labels}')
    # HAC_model = AgglomerativeClustering(n_clusters=None, distance_threshold=0.3, metric='precomputed', linkage='average')
    # labels = HAC_model.fit_predict(1-(0.7*cs_sim_matrix+0.3*cos_sim).cpu())
    # print(f'0.7:0.3 {labels}')
    clusters = {}
    for string, label in zip(mentions, labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(string)
    return clusters


def self_veri(path):
    text_list = []
    if os.path.exists(path):
        llm_text = read_file(path).replace(' ','').split('\n')
        for item in llm_text:
            if len(item)>0:
                text_list.append(item)
        return text_list
    else:
        print(f'Path: {path} No file!')
        return None


def find_string_in_html(html_content, target_string):
    # 解析HTML内容
    soup = BeautifulSoup(html_content, 'html.parser')

    # 初始化结果列表
    results = []

    # 定义递归函数，用于遍历所有标签
    def search_element(element, current_level=-1, char_count=0):
        if isinstance(element, str):  # 如果当前元素是字符串而不是标签
            index = element.find(target_string)
            if index != -1:  # 找到目标字符串
                results.append({
                    'tag': element.parent.name if element.parent else 'None',
                    'level': current_level,
                    'char_count': char_count + index
                })
            if element.isspace(): 
                return 0
            else:
                return len(element)  # 返回当前文本的字符长度
        elif element.name:  # 如果当前元素是标签
            char_len = 0
            for child in element.children:  # 遍历当前标签的所有子元素
                char_len += search_element(child, current_level + 1, char_count + char_len)
            return char_len  # 返回当前标签内文本的总字符长度
        return 0
    # 从根标签开始搜索
    char_total = search_element(soup)
    return results, char_total


def is_abbreviation(mention, domain, args):
    if args.lang == 'en':
        word_list = mention.split()
    else:
        word_list = lazy_pinyin(mention)
    
    initials = ''.join(word[0] for word in word_list)
    
    domain = domain.lower()
    
    # return abbreviation == ''.join(full_pinyin) or abbreviation == initials
    return iterative_lcs_similarity(domain, ''.join(word_list)) == 1 or iterative_lcs_similarity(domain, initials) == 1


def ranking_score(pre_dic, filter_mentions, html, aug, args):
    dis_dic = {}
    ### disambiguation
    if len(filter_mentions)>1:
        # print(ground_truth)
        clusters = disambiguation(filter_mentions)
        
        for label in clusters.keys():
            # print(f'cluster:{label}')
            score = 0
            for name in clusters[label]:
                tmp = 0
                if aug != '无':
                    if is_abbreviation(name, aug, args):
                        tmp += pre_dic[name]
                results, char_total = find_string_in_html(html, name)
                for result in results:
                    if result['tag'] == 'title' or result['tag'] == 'meta' or (result['char_count'] / char_total < 0.1 and result['level'] == 3) or (result['char_count'] / char_total > 0.9 and result['tag'] == 'div'):
                        tmp += pre_dic[name]
                score = score + pre_dic[name] + tmp
            dis_dic['，'.join(clusters[label])] = score
    else:
        for mention in filter_mentions:
            tmp = 0
            if aug != '无':
                if is_abbreviation(mention, aug, args):
                    tmp += pre_dic[mention]
            results, char_total = find_string_in_html(html, mention)
            for result in results:
                if result['tag'] == 'title' or result['tag'] == 'meta' or (result['char_count'] / char_total < 0.1 and result['level'] == 3) or (result['char_count'] / char_total > 0.9 and result['tag'] == 'div'):
                    tmp += pre_dic[mention]
            dis_dic[mention] = pre_dic[mention] + tmp
    
    sorted_list = sorted(dis_dic.items(), key=lambda x: x[1], reverse = True)
    
    return dis_dic, sorted_list


def main(args):
    print(args.res_path)
    aug_path = os.path.join(args.aug_path, 'results')
    self_veri_path = os.path.join(args.res_path, '../confirm/results/')
    
    F_before_sv = 0
    F_after_sv = 0
    for item in tqdm(os.listdir(args.res_path)):
        # print(f'------{item}------')
        with open(os.path.join(args.raw_path, item) + '.html','r',encoding='utf-8') as f:
            html = f.read()
        
        soup = BeautifulSoup(' '.join(html.split('\n')), 'html.parser')
        text = soup.get_text()
        # print(f'raw: {text}')
        
        if args.aug is True:
            aug = read_file(aug_path + item)
            pa = r'^\d+[\.\d+]*$'
            if len(re.findall(pa, aug)) > 0 or aug == '无相关信息':
                aug = '无'
            else:
                aug = aug.replace('{','').replace('}','').replace('\n','')
        # print(f'aug: {aug}')
        
        name = item.split('_')[0]
        label = item.split('.html')[0].replace('\\','/').split('_')[1:]
        
        output_path = os.path.join(args.res_path, item)
        with open(output_path,'r',encoding='utf-8') as f:
            data = f.read()
        # print(f'llm output: {data}')
        
        if args.lang == 'en':
            name_score_dic = get_name_score_list_en(data)
        else:
            name_score_dic = get_name_score_list(data)
        
        if args.verified is False:
            print(f'dealed output: {name_score_dic}')
            dealed_result_path = os.path.join(args.res_path, '../dealed/')
            if not os.path.isdir(dealed_result_path):
                os.makedirs(dealed_result_path)
            write_json(os.path.join(dealed_result_path, item + '.json'), name_score_dic)
            continue
        
        mentions = self_veri(self_veri_path + item)
        filter_mentions = []
        for mention in mentions:
            if len(mention)>0 and mention.lower() in html.lower():
                filter_mentions.append(mention)
        
        flag = False
        for can in name_score_dic.keys():
            for l in label:
                if iterative_lcs_similarity(can, l) != 1:
                    flag = True
        if flag:
            F_before_sv += 1
        flag = False
        for can in filter_mentions:
            for l in label:
                if iterative_lcs_similarity(can, l) != 1:
                    flag = True
        if flag:
            F_after_sv += 1
        
        dis_dic, sorted_list = ranking_score(name_score_dic, filter_mentions, html, aug, args)
        # print(f'disambiguated and sorted output: {sorted_list}')
        ranking_result_path = os.path.join(args.res_path, '../ranking/')
        if not os.path.isdir(ranking_result_path):
            os.makedirs(ranking_result_path)
        write_json(os.path.join(ranking_result_path, item + '.json'), sorted_list)
        
    print(f'F_before_sv: {F_before_sv / len(os.listdir(args.res_path))}  F_confirm: {F_after_sv / len(os.listdir(args.res_path))}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_path', type=str, default='./data/woi_cn/test/')
    parser.add_argument('--res_path', type=str, default='./data/woi_cn/OwnerHunter/results/')
    parser.add_argument('--lang', type=str, default='ch')
    parser.add_argument('--aug', type=bool, default=True)
    parser.add_argument('--verified', type=bool, default=True)
    args = parser.parse_args()
    main(args)