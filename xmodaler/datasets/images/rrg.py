# Copyright 2022 JD.com, Inc., JD AI
"""
@author: Jianjie Luo
@contact: jianjieluo.sysu@gmail.com
"""
import os
import copy
import pickle
import random
import torch
from tqdm import tqdm
import numpy as np
from PIL import Image
from torchvision import transforms
from .mscoco import MSCoCoDataset
from xmodaler.config import configurable
from xmodaler.config import kfg
from xmodaler.functional import read_np, dict_as_tensor
from ..build import DATASETS_REGISTRY

__all__ = ["RRGDiffusionDataset"]


@DATASETS_REGISTRY.register()
class RRGDiffusionDataset:
    @configurable
    def __init__(
            self,
            stage: str,
            anno_folder: str,
            max_seq_len: int,
            image_path: str,
            cas_rand_ratio,
            dataset_name: str
    ):
        self.stage = stage
        self.anno_folder = anno_folder
        self.max_seq_len = max_seq_len
        self.image_path = image_path
        self.cas_rand_ratio = cas_rand_ratio
        if stage == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        self.dataset_name = dataset_name

    @classmethod
    def from_config(cls, cfg, stage: str = "train"):
        ret = {"stage": stage,
               "anno_folder": cfg.DATALOADER.ANNO_FOLDER,
               "max_seq_len": cfg.MODEL.MAX_SEQ_LEN,
               "image_path": cfg.DATALOADER.IMAGE_PATH,
               "cas_rand_ratio": cfg.DATALOADER.CASCADED_SENT_RAND_RATIO,
               'dataset_name': cfg.DATASETS.NAME
               }
        return ret

    def load_data(self, cfg):

        if self.stage == 'test' and cfg.DATALOADER.INFERENCE_TRAIN == True:
            datalist = []
            for split in ['train', 'test']:
                anno_file = self.anno_folder.format(split)
                tmp_datalist = pickle.load(open(anno_file, 'rb'), encoding='bytes')
                datalist.extend(tmp_datalist)
        else:
            datalist = pickle.load(open(self.anno_folder, 'rb'), encoding='bytes')
        datalist = datalist[self.stage]

        if len(cfg.DATALOADER.CASCADED_FILE) > 0:
            cascaded_pred = pickle.load(open(cfg.DATALOADER.CASCADED_FILE, 'rb'), encoding='bytes')
            for i in range(len(datalist)):
                image_id = str(datalist[i]['image_id'])
                datalist[i]['cascaded_tokens_ids'] = cascaded_pred[image_id]

        return datalist

    def __call__(self, dataset_dict):
        dataset_dict = copy.deepcopy(dataset_dict)
        id = dataset_dict['id']
        image_list = dataset_dict['image_path']
        images = []
        for path in image_list:
            image = Image.open(os.path.join(self.image_path, path)).convert('RGB')
            image = self.transform(image)
            images.append(image)

        if self.stage != 'train':
            u_tokens_ids = np.array(dataset_dict['report'], dtype=np.int64)
            u_tokens_type = np.zeros(self.max_seq_len, dtype=np.int64)
        else:
            u_tokens_ids = np.array(dataset_dict['report'], dtype=np.int64)[:self.max_seq_len]
            u_tokens_ids[-1] = 0
            u_tokens_type = np.zeros((len(u_tokens_ids)), dtype=np.int64)
        if self.dataset_name == 'MIMIC_CXR':
            ret = {
                kfg.IDS: id,
                kfg.U_TOKENS_IDS: u_tokens_ids,
                kfg.U_TOKENS_TYPE: u_tokens_type,
                kfg.IMAGES: images,
            }
        else:
            ret = {
                kfg.IDS: id,
                kfg.U_TOKENS_IDS: u_tokens_ids,
                kfg.U_TOKENS_TYPE: u_tokens_type,
                kfg.IMAGES: images,
            }

        if self.stage != 'train':
            dict_as_tensor(ret)
            return ret

        dict_as_tensor(ret)
        return ret
