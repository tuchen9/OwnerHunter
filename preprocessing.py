import os
import random
import shutil
from tqdm import tqdm


def read_file(path):
    with open(path, "r",encoding='utf-8') as f:
        lines = f.readlines()
    return "".join(lines)


FileType = ["json","csv"]
def deal_folder(file_list, path):
    for fl in os.listdir(path):
        now_path = os.path.join(path, fl)
        if os.path.isfile(now_path):
            if now_path.split(".")[-1] not in FileType:
                file_list.append(now_path)


def woi_cn_split():
    file_list = []
    path = './1/1_label_html/'
    deal_folder(file_list, path)
    print(len(file_list))
    path = './2/2_label_html/'
    deal_folder(file_list, path)
    print(len(file_list))
    path = './3/3_label_html/'
    deal_folder(file_list, path)
    print(len(file_list))

    random.shuffle(file_list)

    n = len(file_list)
    n_train = int(0.2 * n)
    n_valid = int(0.2 * n)

    train_files = file_list[:n_train]
    valid_files = file_list[n_train:n_train+n_valid]
    test_files = file_list[n_train+n_valid:]

    base_dir = './woi_cn/'
    # 定义输出目录 
    dirs = {
        "train": os.path.join(base_dir, "train"),
        "valid": os.path.join(base_dir, "valid"),
        "test": os.path.join(base_dir, "test")
    }

    # 创建目录
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)
    
    # 复制文件
    def copy_files(files, target_dir):
        for f in tqdm(files):
            shutil.copy(f, target_dir)

    copy_files(train_files, dirs["train"])
    copy_files(valid_files, dirs["valid"])
    copy_files(test_files, dirs["test"])

    print(f"数据集划分完成：train={len(train_files)}, valid={len(valid_files)}, test={len(test_files)}")


def woi_de_jp_split(path):
    # load data
    file_list = []
    deal_folder(file_list, path)
    print(len(file_list))
    
    random.shuffle(file_list)

    n = len(file_list)
    n_train = int(0.4 * n)
    train_files = file_list[:n_train]
    test_files = file_list[n_train:]

    base_dir = '/'.join(path.split('/')[:-1 ])
    dirs = {
        "train": os.path.join(base_dir, "train"),
        "test": os.path.join(base_dir, "test"),
    }

    # 创建目录
    for d in dirs.values():
        os.makedirs(d, exist_ok=True)

    # 复制文件
    def copy_files(files, target_dir):
        for f in tqdm(files):
            shutil.copy(f, target_dir)

    copy_files(train_files, dirs["train"])
    copy_files(test_files, dirs["test"])

    print(f"数据集划分完成：train={len(train_files)}, test={len(test_files)}")


if __name__ == "__main__":
    woi_cn_split()
    woi_de_jp_split('./de/de_label_html')
    woi_de_jp_split('./jp/jp_label_html')