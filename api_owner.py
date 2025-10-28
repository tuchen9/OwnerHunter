import os
import re
import json
import argparse
import torch
import torch.nn.functional as F
from bs4 import BeautifulSoup
from tqdm import tqdm
from pypinyin import lazy_pinyin
import dashscope
from dashscope import Generation
from dashscope.api_entities.dashscope_response import Role
from openai import OpenAI
from transformers import AutoTokenizer, BertModel
import prompt_generator as pg


API_KEY = "******" #这里填入api的sdk码
dashscope.api_key = API_KEY


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


FileType = ["json","csv"]
def deal_folder(file_list, path):
    for fl in os.listdir(path):
        now_path = os.path.join(path, fl)
        if os.path.isfile(now_path):
            if now_path.split(".")[-1] not in FileType:
                file_list.append(now_path)


def llm_once(prompt,args):
    client = OpenAI(api_key=API_KEY, base_url="https://dashscope.aliyuncs.com/compatible-mode/v1")

    try:
        response = client.chat.completions.create(
            model=args.model,
            messages=[
                {"role": "user", "content": prompt},
            ],
            stream=False,
            temperature=0.01,
            seed=args.seed
        )
        if response.choices:
            # print(response)
            return response.choices[0].message.content
        else:
            print(response)
            return ''
    except Exception as e:
        print(e)
        return ''


def llm_confirm(name_list,args):
    messages = []
    confirm_name_list = []
    for name in name_list:
        messages.append({'role': Role.USER, 'content': pg.get_prompt_filter(name)})
        response = Generation.call(args.model, messages=messages, result_format='message')
        if response.output:
            output = response.output.choices[0]['message']['content']
            #print(output)
            if '否' not in output:
                confirm_name_list.append(name)
            messages.append({'role': response.output.choices[0]['message']['role'], 'content': output})
    return confirm_name_list


def get_name_score_list(output):
    name_score_dic = {}
    output = output.replace('[', '\n').replace(']', '\n').replace('{', '\n').replace('}', '\n').replace(',', '\n').replace('，', '\n').replace('、', '\n').replace('。', '\n').replace('“', '\n').replace('”', '\n').replace('"', '\n').split()
    # print(output)
    pa = r'^[0-9A-Za-z]+$'
    for item in output:
        if '本文' not in item and '文本' not in item and '公网安备' not in item and '号' not in item and '抱歉' not in item and '无法' not in item and '我们' not in item and '例子' not in item and '结果为' not in item and '未找到' not in item and '没有' not in item and '备案' not in item and '所有者' not in item and '根据' not in item and '规则' not in item and item != '无':
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


def main(args, file_list, exists_results, exists_errors):
    tokenizer = AutoTokenizer.from_pretrained('../Qwen-main/Qwen/Qwen-14B-Chat', trust_remote_code=True)

    for error_name in exists_errors:
        if os.path.exists(args.output_path + '/results/' + error_name):
            os.remove(args.output_path + '/results/' + error_name)
        os.remove(args.output_path + '/errors/' + error_name)
    
    aug_path = os.path.join(args.aug_path, 'results')
    example_path = os.path.join(args.input_path, 'examples')

    for filepath in tqdm(file_list):
        name = filepath.split('/')[-1].split('.html')[0]
        if name in exists_results and name not in exists_errors:
            continue
        print('\n-----' + name + '-----')

        website = read_file(filepath)
        text_website = ' '.join(website.split('\n'))

        if 'aug' in args.mode:
            multi_aug = read_file(aug_path + name)
            pa = r'^\d+[\.\d+]*$'
            if len(re.findall(pa, multi_aug)) > 0 or multi_aug == '无相关信息':
                multi_aug = '无'
            else:
                multi_aug = multi_aug.replace('{','').replace('}','').replace('\n','')
            print('***' + multi_aug + '\n')
        else:
            multi_aug = ''

        if 'example' in args.mode:
            example = read_json(example_path + name)[:args.k]
            # print(example)
        
        website = read_file(filepath)
        text_website = ' '.join(website.split('\n'))
        token_list = []

        soup = BeautifulSoup(text_website, 'html.parser')
        text_website = soup.get_text(separator='|b|')
        text_list = text_website.split('|b|')
        space_token = tokenizer.encode(' ')
        token_tmp = []
        count = 1
        for k in range(len(text_list)):
            if len(text_list[k]) > 0 and text_list[k].isspace() == False:
                raw_tokens = tokenizer.encode(' '.join(text_list[k].split()))
                if len(token_tmp) + len(raw_tokens) <= args.token_num:
                    if len(token_tmp) == 0:
                        token_tmp = raw_tokens
                    else:
                        token_tmp = token_tmp + space_token + raw_tokens
                else:
                    token_list.append(token_tmp)
                    tmp = 1 if len(raw_tokens) % args.token_num > 0 else 0
                    for i in range(len(raw_tokens) // args.token_num + tmp):
                        if len(raw_tokens) > args.token_num*(i+1):
                            token_list.append(raw_tokens[args.token_num*i:args.token_num*(i+1)])
                        else:
                            token_tmp = raw_tokens[args.token_num*i:]
                count += 1
            if k == len(text_list)-1:
                token_list.append(token_tmp)
        
        outputs = ''
        for i in range(len(token_list)):
            if args.mode == 'vanilla':
                msg = pg.get_prompt_vanilla(tokenizer.decode(token_list[i]))
            else:
                msg = pg.get_prompt_aug_text_score_example(multi_aug, tokenizer.decode(token_list[i]), example)

            output = llm_once(msg,args)
            print('---' + output)
            if output and len(output)>0:
                outputs = outputs + output + '\n'
                with open(args.output_path + '/results/' + name, 'a') as fresult:
                    fresult.write(output + '\n')
            else:
                with open(args.output_path + '/errors/' + name, 'a') as ferror:
                    ferror.write('error!\n')
        
        if len(outputs)>0:
            if args.lang == 'en':
                name_score_dic = get_name_score_list_en(outputs)
            else:
                name_score_dic = get_name_score_list(outputs)
            confirm_mentions = '，'.join(llm_confirm(name_score_dic.keys(),args))
            print('----' + confirm_mentions + '\n')
            with open(args.output_path + '/confirm/' + name, 'w') as fresult:
                fresult.write(confirm_mentions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen1.5-72b-chat')
    parser.add_argument('--input_path', type=str, default='./data/woi_cn/test', help='input data directory')
    parser.add_argument('--output_path', type=str, default='./data/woi_cn/OwnerHunter', help='output data directory')
    parser.add_argument('--aug_path', type=str, default='./data/woi_cn/OwnerHunter/aug', help='input data directory')
    parser.add_argument('--token_num', type=int, default=30000)
    parser.add_argument('--mode', type=str, default='aug-example')
    parser.add_argument('--lang', type=str, default='ch')
    parser.add_argument("--seed", type=int, default=12, help="Random seed for reproducibility")
    parser.add_argument('--k', type=int, default=1)
    args = parser.parse_args()

    if not os.path.isdir(args.output_path):
        print('Making new dir: ' + args.output_path)
        os.makedirs(args.output_path)
        os.makedirs(args.output_path + '/results')
        os.makedirs(args.output_path + '/errors')
        os.makedirs(args.output_path + '/domain')
        os.makedirs(args.output_path + '/confirm')
    else:
        print('Dir exists: ' + args.output_path)
    
    # check results
    exists_results = set(os.listdir(args.output_path + '/results'))
    print('\nNum of exists results:' + str(len(exists_results)))
    exists_errors = set(os.listdir(args.output_path + '/errors'))
    print('\nNum of exists errors:' + str(len(exists_errors)))
    
    # load data
    file_list = []
    if os.path.isfile(args.input_path):
        if args.input_path.split(".")[-1] not in FileType:
            file_list.append(args.input_path)
    elif os.path.isdir(args.input_path):
        deal_folder(file_list, args.input_path)
    else:
        print("Please specify a correct path!")

    main(args, file_list, exists_results, exists_errors)