"""
Usage:
python3 -m fastchat.serve.huggingface_api --model ~/model_weights/vicuna-7b/
"""
import os
import re
import json
import resource
import argparse
import torch
from bs4 import BeautifulSoup
from tqdm import tqdm
from transformers import set_seed

from fastchat.model import load_model, get_conversation_template, add_model_args
from fastchat.src_all.prompt_generator import PromptGenerator


def set_memory_limit(maxsize):
    _, hard = resource.getrlimit(resource.RLIMIT_AS)
    if maxsize is None:
        maxsize = hard
    resource.setrlimit(resource.RLIMIT_AS, (maxsize, hard))


def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = json.load(f)
    return content


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
        # elif os.path.isdir(now_path):
        #     deal_folder(file_list, now_path, args)
        # else:
        #     print("Please specify a correct path!")


@torch.inference_mode()
def main(args, generator, file_list, exists_results, exists_errors):
    model, tokenizer = load_model(
        args.model_path,
        device=args.device,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
        load_8bit=args.load_8bit,
        cpu_offloading=args.cpu_offloading,
        revision=args.revision,
        debug=args.debug,  
    )
    
    for error_name in exists_errors:
        if os.path.exists(args.output_path + '/results/' + error_name):
            os.remove(args.output_path + '/results/' + error_name)
        os.remove(args.output_path + '/errors/' + error_name)
    
    aug_path = args.input_path + '/multi_aug/results/'
    example_path = args.input_path + '/examples/'
    for filepath in tqdm(file_list):
        name = filepath.split('/')[-1].split('.html')[0]
        if name in exists_results and name not in exists_errors:
            continue
        print('\n-----' + name + '-----')
        
        if 'aug' in args.mode:
            multi_aug = read_file(aug_path + name)
            pa = r'^\d+[\.\d+]*$'
            if len(re.findall(pa, multi_aug)) > 0 or multi_aug == '无相关信息':
                multi_aug = '无'
            else:
                multi_aug = multi_aug.replace('{','').replace('}','').replace('\n','')
            print('***' + multi_aug + '\n')
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
                if args.mode == 'block':
                    raw_tokens = tokenizer.encode('[' + str(count) + ']' + ' '.join(text_list[k].split()))
                else:
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
        
        
        for i in range(len(token_list)):
            if args.mode == 'simple':
                msg = generator.get_prompt_simple(tokenizer.decode(token_list[i]))
            else:
                msg = generator.get_prompt_aug_text_score_example(multi_aug, tokenizer.decode(token_list[i]), example)

            try:
                conv = get_conversation_template(args.model_path).copy()
                conv.append_message(conv.roles[0], msg)
                conv.append_message(conv.roles[1], None)
                prompt = conv.get_prompt()

                inputs = tokenizer([prompt], return_tensors="pt").to(args.device)
                output_ids = model.generate(
                    **inputs,
                    do_sample=True if args.temperature > 1e-5 else False,
                    temperature=args.temperature,
                    repetition_penalty=args.repetition_penalty,
                    max_new_tokens=args.max_new_tokens,
                )
                if model.config.is_encoder_decoder:
                    output_ids = output_ids[0]
                else:
                    output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
                outputs = tokenizer.decode(
                    output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
                )
                
                print('---' + outputs)
                with open(args.output_path + '/results/' + name, 'a') as fresult:
                    fresult.write(outputs)

            except Exception as e:
                with open(args.output_path + '/errors/' + name, 'a') as ferror:
                    ferror.write('\n---\n')
                    ferror.write(str(e))


if __name__ == "__main__":
    set_memory_limit(220 * 1024 * 1024 * 1024)
    
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument(
        "--conv-template", type=str, default=None, help="Conversation prompt template."
    )
    parser.add_argument("--temperature", type=float, default=0.01)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--message", type=str, default="Hello! Who are you?")

    parser.add_argument('--input_path', type=str, default='./data/', help='input data directory')
    parser.add_argument('--output_path', type=str, default='./data/OwnerHunter/', help='output data directory')
    parser.add_argument('--mode', type=str, default='text')
    parser.add_argument("--seed", type=int, default=12, help="Random seed for reproducibility")
    parser.add_argument('--token_num', type=int, default=1500)
    parser.add_argument('--k', type=int, default=1)

    args = parser.parse_args()
    
    set_seed(args.seed)
    
    # Reset default repetition penalty for T5 models.
    if "t5" in args.model_path and args.repetition_penalty == 1.0:
        args.repetition_penalty = 1.2


    if not os.path.isdir(args.output_path):
        print('Making new dir: ' + args.output_path)
        os.makedirs(args.output_path)
        os.makedirs(args.output_path + '/results')
        os.makedirs(args.output_path + '/errors')
    else:
        print('Dir exists: ' + args.output_path)
    
    # check results
    exists_results = set(os.listdir(args.output_path + '/results'))
    print('\nNum of exists results:' + str(len(exists_results)))
    exists_errors = set(os.listdir(args.output_path + '/errors'))
    print('\nNum of exists errors:' + str(len(exists_errors)))
    
    # load data
    FileType = ["json","csv"]
    file_list = []
    if os.path.isfile(args.input_path):
        if args.input_path.split(".")[-1] not in FileType:
            file_list.append(args.input_path)
    elif os.path.isdir(args.input_path):
        deal_folder(file_list, args.input_path, FileType)
    else:
        print("Please specify a correct path!")

    generator = PromptGenerator(args.model_path)

    main(args, generator, file_list, exists_results, exists_errors)
