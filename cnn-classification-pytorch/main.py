import os
import argparse
import datetime
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader

# Paramters设定
parser = argparse.ArgumentParser(description='CNN text classificer')
# learning
parser.add_argument('-lr', type=float, default=0.001, help='initial learning rate [default: 0.001]')
parser.add_argument('-epochs', type=int, default=256, help='number of epochs for train [default: 256]')
parser.add_argument('-batch-size', type=int, default=64, help='batch size for training [default: 64]')
parser.add_argument('-log-interval',  type=int, default=1,   help='how many steps to wait before logging training status [default: 1]')
parser.add_argument('-test-interval', type=int, default=100, help='how many steps to wait before testing [default: 100]')
parser.add_argument('-save-interval', type=int, default=500, help='how many steps to wait before saving [default:500]')
parser.add_argument('-save-dir', type=str, default='snapshot', help='where to save the snapshot')
parser.add_argument('-early-stop', type=int, default=1000, help='iteration numbers to stop without performance increasing')
parser.add_argument('-save-best', type=bool, default=True, help='whether to save when get best performance')
# data
parser.add_argument('-shuffle', action='store_true', default=False, help='shuffle the data every epoch')
# model
parser.add_argument('-dropout', type=float, default=0.5, help='the probability for dropout [default: 0.5]')
parser.add_argument('-max-norm', type=float, default=3.0, help='l2 constraint of parameters [default: 3.0]')
parser.add_argument('-embed-dim', type=int, default=128, help='number of embedding dimension [default: 128]')
parser.add_argument('-kernel-num', type=int, default=100, help='number of each kind of kernel')
parser.add_argument('-kernel-sizes', type=str, default='3,4,5', help='comma-separated kernel size to use for convolution')
parser.add_argument('-static', action='store_true', default=False, help='fix the embedding')
# device
parser.add_argument('-device', type=int, default=-1, help='device to use for iterate data, -1 mean cpu [default: -1]')
parser.add_argument('-no-cuda', action='store_true', default=False, help='disable the gpu')
# option
parser.add_argument('-snapshot', type=str, default=None, help='filename of model snapshot [default: None]')
parser.add_argument('-predict', type=str, default=None, help='predict the sentence given')
parser.add_argument('-test', action='store_true', default=False, help='train or test')
parser.add_argument('-datasets-dir', type=str, default='data', help='train or test')
args = parser.parse_args()


# update args and print
args.cuda = (not args.no_cuda) and torch.cuda.is_available();del args.no_cuda
args.kernel_sizes = [int(k) for k in args.kernel_sizes.split(',')]
args.save_dir = os.path.join(args.save_dir, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
os.environ["HF_DATASETS_CACHE"]=args.datasets_dir
tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")


# data preprocessing
def tokenize_function(examples):
    return tokenizer(examples["sentence"], padding="max_length", truncation=True, max_length=max_length)


def mr(batch_size):
    # download MR dataset
    mr_dataset = load_dataset("imdb")

    mr_dataset = mr_dataset.map(tokenize_function, batched=True)

    # create DataLoader
    train_iter = mr_dataset["train"].shuffle().select(range(20000)).to_dataloader(batch_size=batch_size, shuffle=True)
    dev_iter = mr_dataset["test"].to_dataloader(batch_size=batch_size)  # 通常使用测试集作为验证集

    return train_iter, dev_iter


def sst(batch_size):
    # download MR dataset
    sst_dataset = load_dataset("sst2")

    sst_dataset = sst_dataset.map(tokenize_function, batched=True)

    # create DataLoader
    train_iter = sst_dataset["train"].shuffle().select(range(10000)).to_dataloader(batch_size=batch_size, shuffle=True)
    dev_iter = sst_dataset["validation"].to_dataloader(batch_size=batch_size)
    test_iter = sst_dataset["test"].to_dataloader(batch_size=batch_size)

    return train_iter, dev_iter, test_iter


dataset = load_dataset("imdb")
# 创建自定义拆分
train_test_dataset = dataset["train"].train_test_split(test_size=0.2)
print(train_test_dataset)
    
