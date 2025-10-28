import os
import json
import argparse
import torch
from tqdm import tqdm
from transformers import set_seed

from fastchat.model import load_model, get_conversation_template, add_model_args
from fastchat.src_all.prompt_generator import PromptGenerator


def read_file(path):
    with open(path, "r",encoding='utf-8') as f:
        lines = f.readlines()
    return "".join(lines)

def write_file(path, context):
    with open(path,'w') as f:
        f.write(context)

def read_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        content = json.load(f)
    return content

def deal_folder(file_list, path, args):
    for fl in os.listdir(path):
        now_path = os.path.join(path, fl)
        if os.path.isfile(now_path):
            file_list.append(now_path)
        elif os.path.isdir(now_path):
            deal_folder(file_list, now_path, args)
        else:
            print("Please specify a correct path!")


@torch.inference_mode()
def main(args, generator, file_list, exists_results, exists_errors, output_path):
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
        if os.path.exists(output_path + '/results/' + error_name):
            os.remove(output_path + '/results/' + error_name)
        os.remove(output_path + '/errors/' + error_name)
    
    for filepath in tqdm(file_list):
        name = '.'.join(filepath.split('/')[-1].split('.')[:-1])
        if name in exists_results and name not in exists_errors:
            continue
        print('\n-----' + name + '-----')

        dealed_json = read_json(filepath)
        
        for text in dealed_json.keys():
            print(text)
            try:
                msg = generator.get_prompt_filter(text)
                
                # print(msg)
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
                if '否' not in outputs:
                    with open(output_path + '/results/' + name, 'a') as fresult:
                        fresult.write(text+'\n')

            except Exception as e:
                with open(output_path + '/errors/' + name, 'a') as ferror:
                    ferror.write('\n---\n')
                    ferror.write(str(e))


if __name__ == "__main__":
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

    parser.add_argument('--input_path', type=str, default='..data/OwnerHunter/dealed/', help='input data directory')
    parser.add_argument("--seed", type=int, default=12, help="Random seed for reproducibility")
    parser.add_argument('--token_num', type=int, default=1500)

    args = parser.parse_args()
    
    set_seed(args.seed)
    
    output_path = os.path.join(args.input_path, f'../confirm/')
    if not os.path.isdir(output_path):
        print('Making new dir: ' + output_path)
        os.makedirs(output_path)
        os.makedirs(output_path + '/results')
        os.makedirs(output_path + '/errors')
    else:
        print('Dir exists: ' + output_path)
    
    # check results
    exists_results = set(os.listdir(output_path + '/results'))
    print('\nNum of exists results:' + str(len(exists_results)))
    exists_errors = set(os.listdir(output_path + '/errors'))
    print('\nNum of exists errors:' + str(len(exists_errors)))
    
    # load data
    file_list = []
    if os.path.isfile(args.input_path):
        file_list.append(args.input_path)
    elif os.path.isdir(args.input_path):
        deal_folder(file_list, args.input_path, args)
    else:
        print("Please specify a correct path!")

    generator = PromptGenerator(args.model_path)

    main(args, generator, file_list, exists_results, exists_errors, output_path)
