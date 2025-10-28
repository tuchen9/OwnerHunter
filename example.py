import os
import re
import sys
import json
import glob
import argparse
import torch
import torch.nn.functional as F
from bs4 import BeautifulSoup
from tqdm import tqdm
from transformers import AutoTokenizer, BertModel


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = json.load(f)
    return content

def write_json(path, context):
    with open(path,'w') as f:
        json.dump(context,f)

def read_file(path):
    with open(path, "r",encoding='utf-8') as f:
        lines = f.readlines()
    return "".join(lines)

def write_file(path, context):
    with open(path,'w') as f:
        f.write(context)

def deal_folder(file_list, path, FileType):
    for fl in os.listdir(path):
        now_path = os.path.join(path, fl)
        if os.path.isfile(now_path):
            if now_path.split(".")[-1] not in FileType:
                file_list.append(now_path)


def split(tokenizer, text, max_token_len=510):
    batch_sentences = []
    raw_token = tokenizer.encode(text, add_special_tokens=False)
    tmp = 1 if len(raw_token) % max_token_len > 0 else 0
    for i in range(len(raw_token) // max_token_len + tmp):
        if len(raw_token) > max_token_len*(i+1):
            batch_sentences.append(tokenizer.decode(raw_token[max_token_len*i:max_token_len*(i+1)]))
        else:
            batch_sentences.append(tokenizer.decode(raw_token[max_token_len*i:]))
    return batch_sentences


def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device(args.device)

    train_file_list = []
    FileType = ['json','csv']
    train_path = os.path.join('/'.join(args.input_path.split('/')[:-1]), 'train')
    deal_folder(train_file_list, train_path, FileType)
    print(f"加载示例池{len(train_file_list)}条")
    
    aug_path = os.path.join(train_path,'aug', 'results')
    bert_tokenizer = AutoTokenizer.from_pretrained(args.bert_path)
    bert_model = BertModel.from_pretrained(args.bert_path).to(device)
    examples = []
    example_embeddings = []
    pa = r'^\d+[\.\d+]*$'
    
    with torch.no_grad():
        for filepath in tqdm(train_file_list):
            name = filepath.split('/')[-1].split('.html')[0]
            id = name.split('_')[0]
            label = ','.join(name.split('_')[1:])
            website = ' '.join(read_file(filepath).split('\n'))
            soup = BeautifulSoup(website, 'html.parser')
            text_website = soup.get_text(separator=' ')
            if len(text_website)>50 and len(text_website)<1000:
                item={}
                item['page_text']=text_website
                item['owner_name']=label
                file = glob.glob(aug_path + id +'*')[0]
                example_aug = read_file(file)
                if len(re.findall(pa, example_aug)) > 0:
                    example_aug = '无'
                else:
                    example_aug = example_aug.replace('{','').replace('}','').replace('\n','')
                item['aug']=example_aug
                examples.append(item)
                sentences = split(bert_tokenizer, item['page_text'], 510)
                batch = bert_tokenizer(sentences, max_length=512, truncation=True, padding='max_length', return_tensors="pt").to(device)
                outputs = bert_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], token_type_ids=batch['token_type_ids'])
                last_hidden_state = outputs.last_hidden_state  # [B, L, H]
                attention_mask = batch['attention_mask'].unsqueeze(-1)  # [B, L, 1]
                sum_embeddings = (last_hidden_state * attention_mask).sum(dim=1)  # [B, H]
                sum_mask = attention_mask.sum(dim=1).clamp(min=1e-9)  # [B, 1]
                sentence_embeddings = sum_embeddings / sum_mask  # [B, H]
                example_embeddings.append(torch.mean(sentence_embeddings, dim=0))
        example_embeddings = F.normalize(torch.stack(example_embeddings).to(device), dim=-1).permute(1,0)

    print(f"筛选得到示例{len(examples)}条")
    
    # load data
    file_list = []
    deal_folder(file_list, args.input_path, FileType)
    
    for filepath in tqdm(file_list):
        name = filepath.split('/')[-1].split('.html')[0]
        id = name.split('_')[0]
        website = ' '.join(read_file(filepath).split('\n'))
        soup = BeautifulSoup(website, 'html.parser')
        text_website = soup.get_text(separator=' ')
        sentences = split(bert_tokenizer, text_website, 510)[:10]
        if len(sentences)==0:
            print("None")
            sentences = split(bert_tokenizer, website, 510)[:10]
        with torch.no_grad():
            batch = bert_tokenizer(sentences, max_length=512, truncation=True, padding='max_length', return_tensors="pt").to(device)
            outputs = bert_model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], token_type_ids=batch['token_type_ids'])
            last_hidden_state = outputs.last_hidden_state  # [B, L, H]
            attention_mask = batch['attention_mask'].unsqueeze(-1)  # [B, L, 1]
            sum_embeddings = (last_hidden_state * attention_mask).sum(dim=1)  # [B, H]
            sum_mask = attention_mask.sum(dim=1).clamp(min=1e-9)  # [B, 1]
            sentence_embeddings = sum_embeddings / sum_mask  # [B, H]
            embedding = torch.mean(sentence_embeddings, keepdim=True, dim=0)
            cos_sim = torch.mm(F.normalize(embedding, dim=-1), example_embeddings).squeeze()
            sorted_cos_sim, sorted_indices = torch.sort(cos_sim, descending=True)
            # print(sorted_cos_sim)
            # print(sorted_indices)
        examples_res = []
        for i in range(args.k):
            examples_res.append({'aug':examples[sorted_indices[i]]['aug'], 'page_text':examples[sorted_indices[i]]['page_text'], 'owner_name':examples[sorted_indices[i]]['owner_name']})
            output_path = args.input_path + '/examples/' + name
            write_json(output_path,examples_res)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # experimental settings
    parser.add_argument("-g", "--gpu", default="1,0", type=str, help="which gpu to use")
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--input_path', type=str, default='./data/woi_cn/test', help='input data directory')
    parser.add_argument('--bert_path', type=str, default='./RoBERTa_wwm_ext/')
    parser.add_argument('--k', type=int, default=5)
    args = parser.parse_args()

    if not os.path.isdir(args.input_path + '/examples/'):
        print('Making new dir: ' + args.input_path + '/examples/')
        os.makedirs(args.input_path + '/examples/')
    main(args)