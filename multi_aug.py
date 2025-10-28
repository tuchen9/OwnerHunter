import os
import argparse
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from PIL import Image


def load_model(model_path="."):
    print("Loading MLLM...")
    
    tokenizer = AutoTokenizer.from_pretrained(
        model_path,
        trust_remote_code=True
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).eval()
    
    if torch.cuda.is_available():
        model = model.cuda()
    
    return model, tokenizer


def get_prompt_aug(domain):

    rules = [
        "1.注意该字符串可能是网站所有者的拼音全称、拼音首字母缩写、英文全称、英文首字母缩写，不包含数字；",
        "2.识别的字符串必须从网站的域名或Logo中提取；",
        "3.返回的信息只需要字符串，不必说明识别的原因；",
        "4.将输出的字符串使用大括号括起来；"
    ]
    prompt_rules = f'现在有一个任务，需要你来协助：根据以下规则从网站的域名和Logo中识别出可能与网站所有者有关的字符串：\n' + '\n'.join(rules)

    prompt_example = "接下来举个例子供你参考，例如，给你以下网站的域名：\n" + \
        "tc441.ustc.edu.cn\n" +\
        "\n按规则输出可能与网站所有者有关的字符串，示例的识别结果为: \n" +\
        "{ustc}\n"

    prompt_instruction = '现在，需要你识别的网站域名为：\n' + domain + '，Logo如图所示。\n\n按上述规则识别可能与网站所有者有关的字符串，识别结果为：'

    return '\n'.join([prompt_rules, prompt_example, prompt_instruction])


def generate_text(args, model, tokenizer, domain, image_path, device="cuda"):
    
    image = Image.open(image_path).convert('RGB')
    
    query = get_prompt_aug(domain)
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "image": image, "content": query}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True
    )

    inputs = inputs.to(device)
    
    # settings
    gen_kwargs = {
        "max_length": 2500, 
        "do_sample": True,
        "top_p": 1,
        "temperature": args.temperature
    }
    
    # 生成图像描述
    with torch.no_grad():  # 禁用梯度计算，节省显存
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return text


FileType = ["html"]
def deal_folder(file_list, path, args):
    for fl in os.listdir(path):
        now_path = os.path.join(path, fl)
        if os.path.isfile(now_path):
            if now_path.split(".")[-1] in FileType:
                file_list.append(now_path)


@torch.inference_mode()
def main(args, file_list, exists_results, exists_errors):
    model, tokenizer = load_model(args.model_path)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    for error_name in exists_errors:
        if os.path.exists(args.output_path + '/results/' + error_name):
            os.remove(args.output_path + '/results/' + error_name)
        os.remove(args.output_path + '/errors/' + error_name)
    
    for filepath in tqdm(file_list):
        name = '.'.join(filepath.split('/')[-1].split('.')[:-1])
        if name in exists_results and name not in exists_errors:
            continue
        print('\n-----' + name + '-----')

        domain = name.split('-')[0]
        logo_path = os.path.join(args.input_path,'imgs',filepath.split('/')[-1])
        
        try:
            outputs = generate_text(args, model, tokenizer, domain, logo_path, device)
            print('---' + outputs)
            with open(args.output_path + '/results/' + name, 'a') as fresult:
                fresult.write(outputs)

        except Exception as e:
            with open(args.output_path + '/errors/' + name, 'a') as ferror:
                ferror.write('\n---\n')
                ferror.write(str(e))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=float, default=0.001)
    parser.add_argument('--model_path', type=str, default='./GLM-4V/', help='model directory')
    parser.add_argument('--input_path', type=str, default='./data/', help='input data directory')
    parser.add_argument('--output_path', type=str, default='./data/OwnerHunter/aug', help='output data directory')
    parser.add_argument("--seed", type=int, default=12, help="Random seed for reproducibility")
    
    args = parser.parse_args()
    
    set_seed(args.seed)
    
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
    file_list = []
    if os.path.isfile(args.input_path):
        if args.input_path.split(".")[-1] in FileType:
            file_list.append(args.input_path)
    elif os.path.isdir(args.input_path):
        deal_folder(file_list, args.input_path, args)
    else:
        print("Please specify a correct path!")

    main(args, file_list, exists_results, exists_errors)
