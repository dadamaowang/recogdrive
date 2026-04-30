# -*- encoding: utf-8 -*-
'''
@File    :   run_training_negdrive.py
@Time    :   2026/01/03 19:47:39
@Author  :   Nuoqian Xiao
@Version :   0.0.1
@Contact :   feimaoxiaotianshi@outlook.com
@License :   (C)Copyright 2024-2025, Nuoqian Xiao
@Status  :    GPU DEBUG
@Desc    :   None
'''

from typing import Tuple, List, Dict 
from pathlib import Path
import logging
import os
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
 
import torch
from torch.utils.data import DataLoader
import torch.distributed as dist

import pytorch_lightning as pl

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataclasses import SceneFilter
from navsim.common.dataloader import SceneLoader
from navsim.planning.training.dataset import Dataset
from navsim.planning.training.agent_lightning_module import AgentLightningVLMRL, VRAMMonitor


logger = logging.getLogger(__name__)

CONFIG_PATH = "config/training"
CONFIG_NAME = "default_training"   


def custom_collate_fn(
        batch: List[
            Tuple[
                Dict[str, torch.Tensor],    # features TODO 使用数据的时候看一眼是啥东西
                Dict[str, torch.Tensor],    # targets 
                str     # tokens/prompt
                ]]
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], List[str]]:

    features_list, targets_list, tokens_list = zip(*batch)

    # extract features 
    history_trajectory = torch.stack([features['history_trajectory'] for features in features_list], dim=0).cpu()
    high_command_one_hot = torch.stack([features['high_command_one_hot'] for features in features_list], dim=0).cpu()
    status_feature = torch.stack([features['status_feature'] for features in features_list], dim=0).cpu()   

    # # TODO 这个训 VLM 的时候暂时不开；
    # last_hidden_state = rnn_utils.pad_sequence(   # pad to same length   
    #     [features['last_hidden_state'] for features in features_list],
    #     batch_first=True,
    #     padding_value=0.0
    # ).clone().detach()
    # # print('last_hidden_state')
    # # print(last_hidden_state)

    # extrace image_path_tensor # TODO 
    image_path_tensor = torch.stack([features['image_path_tensor'] for features in features_list], dim=0).cpu() 

    # extract targets
    trajectory = torch.stack([targets['trajectory'] for targets in targets_list], dim=0).cpu()

    features = {
        'history_trajectory': history_trajectory,
        'high_command_one_hot': high_command_one_hot,
        'status_feature': status_feature,
        # 'last_hidden_state': last_hidden_state,
        'image_path_tensor': image_path_tensor
    }
    targets = {
        'trajectory': trajectory
    }

    return features, targets, tokens_list



# def custom_collate_fn(        # TODO 不 cache 的写个新的
#         batch: List[
#             Tuple[
#                 Dict[str, torch.Tensor],    # features TODO 使用数据的时候看一眼是啥东西
#                 Dict[str, torch.Tensor],    # targets 
#                 str     # tokens/prompt
#                 ]]
#     ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor], List[str]]:

#     features_list, targets_list, tokens_list = zip(*batch)

#     # print('features_list:')
#     # print(features_list)

#     # print('targets_list:')
#     # print(targets_list)

#     # print('tokens_list')
#     # print(tokens_list)

#     # extract features 
#     history_trajectory = torch.stack([features['history_trajectory'] for features in features_list], dim=0).cpu()
#     # print('history_trajectory')
#     # print(history_trajectory)

#     high_command_one_hot = torch.stack([features['high_command_one_hot'] for features in features_list], dim=0).cpu()
#     # print('high_command_one_hot')
#     # print(high_command_one_hot)

#     status_feature = torch.stack([features['status_feature'] for features in features_list], dim=0).cpu()   
#     # print('status_feature')
#     # print(status_feature)

#     # # TODO 这个训 VLM 的时候暂时不开；
#     # last_hidden_state = rnn_utils.pad_sequence(   # pad to same length   
#     #     [features['last_hidden_state'] for features in features_list],
#     #     batch_first=True,
#     #     padding_value=0.0
#     # ).clone().detach()
#     # # print('last_hidden_state')
#     # # print(last_hidden_state)
 
#     # extract targets
#     trajectory = torch.stack([targets['trajectory'] for targets in targets_list], dim=0).cpu()
#     # print('trajectory')
#     # print(trajectory)

#     features = {
#         'history_trajectory': history_trajectory,
#         'high_command_one_hot': high_command_one_hot,
#         'status_feature': status_feature,
#         # 'last_hidden_state': last_hidden_state
#     }
#     targets = {
#         'trajectory': trajectory
#     }

#     return features, targets, tokens_list



def build_datasets(cfg: DictConfig, agent: AbstractAgent) -> Tuple[Dataset, Dataset]:
    """
    Builds training and validation datasets from omega config 
    :param cfg: omegaconf dictionary
    :param agent: interface of agents in NAVSIM
    :return: tuple for training and validation dataset
    """
    train_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)

    """
    SceneFilter 目前用到的 cfg：
    /root/recogdrive/navsim/planning/script/config/common/train_test_split/scene_filter/navtrain.yaml

    其中，train_logs, val_logs 划分：
    /root/recogdrive/navsim/planning/script/config/training/default_train_val_test_log_split.yaml

    """

    if train_scene_filter.log_names is not None:
        train_scene_filter.log_names = [
            log_name for log_name in train_scene_filter.log_names if log_name in cfg.train_logs
        ]
    else:
        train_scene_filter.log_names = cfg.train_logs
    
    val_scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    if val_scene_filter.log_names is not None:
        val_scene_filter.log_names = [log_name for log_name in val_scene_filter.log_names if log_name in cfg.val_logs]
    else:
        val_scene_filter.log_names = cfg.val_logs

    data_path = Path(cfg.navsim_log_path)
    sensor_blobs_path = Path(cfg.sensor_blobs_path)

    train_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=train_scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=True       # NOTE 如果训 VLM, 这里需要设置成 True; 
    )

    val_scene_loader = SceneLoader(
        sensor_blobs_path=sensor_blobs_path,
        data_path=data_path,
        scene_filter=val_scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=True
    )

    train_data = Dataset(
        scene_loader=train_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=None,    # VLM fine-tine must None
        force_cache_computation=False,  # VLM fine-tine must False
    )   

    val_data = Dataset(
        scene_loader=val_scene_loader,
        feature_builders=agent.get_feature_builders(),
        target_builders=agent.get_target_builders(),
        cache_path=None,
        force_cache_computation=False,
    )

    print("数据集检查：")
    print(f"训练数据集大小: {len(train_data)}")
    print(f"验证数据集大小: {len(val_data)}")

    return train_data, val_data



@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for training NegDrive Agent 
    :param cfg: omegaconf dictionary
    """
    local_rank = int(os.getenv('LOCAL_RANK', 0))
    world_size = int(os.getenv('WORLD_SIZE', 1))
    rank = int(os.getenv('RANK', 0))

    dist.init_process_group(
        backend='nccl',
        world_size=world_size,
        rank=rank,
    )
    torch.cuda.set_device(local_rank)
    pl.seed_everything(cfg.seed, workers=True)
    logger.info(f"Global Seed set to {cfg.seed}")

    logger.info(f"Path where all results are stored: {cfg.output_dir}")

    logger.info("Building Agent")
    agent: AbstractAgent = instantiate(cfg.agent) 

    logger.info("Building Lightning Module")    
    lightning_module = AgentLightningVLMRL(
        agent=agent,
    )

    logger.info("Building SceneLoader and Dataset...")
    train_data, val_data = build_datasets(cfg, agent)


    # DOING

    logger.info("Building DataLoader")
    train_dataloader = DataLoader(
        train_data,
        collate_fn=custom_collate_fn, 
        **cfg.dataloader.params,
        shuffle=True
    )
    logger.info("Num training samples: %d", len(train_data))
    val_dataloader = DataLoader(
        val_data,
        collate_fn=custom_collate_fn,
        **cfg.dataloader.params,
        shuffle=False
    )
    logger.info("Num validation samples: %d", len(val_data))


    logger.info("Building Trainer")
    trainer = pl.Trainer(
        **cfg.trainer.params,    
        enable_checkpointing=False,
        # callbacks=[VRAMMonitor()]
        # fast_dev_run=True
    )
    # logger.info("Building Trainer")
    # trainer = pl.Trainer(
    #     **cfg.trainer.params,   # TODO 
    #     callbacks=[pl.callbacks.ModelCheckpoint
    #                (monitor="val/loss_epoch",mode='min', save_top_k=5,every_n_epochs=1)])
    #     # callbacks: Train normally, but also run this checkpoint-saving logic during training.


    logger.info("Starting Training")
    trainer.fit(
        model=lightning_module,
        train_dataloaders=train_dataloader,
        val_dataloaders=val_dataloader,
    )


if __name__ == "__main__":
    main()
