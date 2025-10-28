from transformers import AutoTokenizer

class PromptGenerator:
    def __init__(self, model_path):
        self.model_path = model_path


    def num_tokens_from_string(self, text):
    # Returns the number of tokens in a text string.
        tokenizer = AutoTokenizer.from_pretrained(self.model_path, trust_remote_code=True)
        num_tokens = len(tokenizer.encode(text))
        return num_tokens


    def get_prompt_aug(self, domain):

        rules = [
            "1.注意该字符串可能是网站所有者的拼音全称、拼音首字母缩写、英文全称、英文首字母缩写，不包含数字；",
            "2.识别的字符串必须从网站的域名中提取；",
            "3.返回的信息只需要字符串，不必说明识别的原因；",
            "4.将输出的字符串使用大括号括起来；"
        ]
        prompt_rules = f'现在有一个任务，需要你来协助：根据以下规则从网站的域名中识别出可能与网站所有者有关的字符串：\n' + '\n'.join(rules)

        prompt_example = "接下来列举三个例子供你参考，例如，给你以下网站的域名：\n" + \
            "www.ahguohuagroup.com\n" +\
            "\n按规则输出可能与网站所有者有关的字符串，示例的识别结果为: \n" +\
            "{ahguohuagroup}\n" + \
            "给你以下网站的域名：\n" + \
            "tc441.ustc.edu.cn\n" +\
            "\n按规则输出可能与网站所有者有关的字符串，示例的识别结果为: \n" +\
            "{ustc}\n" + \
            "给你以下网站的域名：\n" + \
            "shyb18.com\n" +\
            "\n按规则输出可能与网站所有者有关的字符串，示例的识别结果为: \n" +\
            "{shyb}\n"

        prompt_instruction = '现在，需要你识别的网站域名为：\n' + domain + '\n\n按上述规则识别可能与网站所有者有关的字符串，识别结果为：'

        return '\n'.join([prompt_rules, prompt_example, prompt_instruction])


    def get_prompt_vanilla(self, website):
        prompt_rules = '请你从网站页面的文本内容中识别出网站的所有者名称，输出格式为{可能的网站所有者名称}，不要输出其他信息，不必说明识别的原因：\n'

        prompt_instruction = '网站文本为：\n' + website + '\n\n识别结果为：'

        return '\n'.join([prompt_rules, prompt_instruction])


    def get_prompt_aug_text_score_example(self, domain, website, examples):

        rules = [
            "1.注意网站的所有者指拥有网站的一个人或组织；",
            "2.网站域名可能是网站的所有者的拼音全称、拼音首字母缩写、英文全称、英文首字母缩写，也可能与所有者无关，在识别所有者时需要灵活参考；",
            "3.识别的所有者名称必须从网站页面文本内容中提取，提取时请考虑文本中的相关上下文，如“版权所有”、“由XXX公司运营”等提示；",
            "4.可能性评分范围为0到1，分数越高表示该名称越有可能是网站的所有者；",
            "5.识别结果的输出格式为{可能的网站所有者名称-可能性评分}，不要输出其他信息，不必说明识别的原因；",
            "6.若有多个可能的所有者名称，则将这些名称都提取出来，并给出每个名称的可能性评分，用逗号隔开每个结果；"
        ]
        prompt_rules = f'现在有一个任务，需要你来协助：根据以下规则结合网站域名从网站页面的文本内容中识别出可能的“网站所有者名称”，并根据其作为所有者的可能性进行评分：\n' + '\n'.join(rules)

        prompt_example = f"接下来列举{len(examples)}个例子供你参考：\n" 
        
        for example in examples:
            prompt_example = prompt_example + "例如，给你网站的域名：" + example['domain'] + "，及页面文本：\n" + example['page_text'] + "\n" +\
            "\n按规则识别可能的网站所有者名称并给出可能性评分，示例的识别结果为: \n" +\
            "{" + example['owner_name'] + "-1}\n"

        prompt_instruction = '现在，需要你识别的网站的域名为：' + domain + '，页面文本为：\n' + website + '\n\n按上述规则识别可能的网站所有者名称并给出可能性评分，{可能的网站所有者名称-可能性评分}，不要输出其他信息，结果为：'

        return '\n'.join([prompt_rules, prompt_example, prompt_instruction])


    def get_prompt_filter(self, text):

        prompt_rules = f'现在有一个任务，需要你来协助：给你一个词，请你判断这个词是不是某个组织或人的名称，输出是或否，不要输出其他信息，不必说明原因：\n'

        prompt_example = "接下来列举两个例子供你参考，例如，给你以下词：\n" + \
            "六安市路安包装制品有限公司\n" +\
            "示例的输出结果为: 是\n" +\
            "给你以下词：\n" + \
            "安徽省建设工程计价\n" +\
            "示例的输出结果为: 否\n"

        prompt_instruction = '现在，需要你判断的词为：\n' + text + '\n输出结果为：'

        return '\n'.join([prompt_rules, prompt_example, prompt_instruction])
