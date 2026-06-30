


# -*- encoding: utf-8 -*-
'''
@File    :   cot_inference.py
@Time    :   2026/06/11 16:28:40
@Author  :   Nuoqian Xiao
@Version :   0.0.1
@Contact :   feimaoxiaotianshi@outlook.com
@License :   (C)Copyright 2024-2025, Nuoqian Xiao
@Status  :   
@Desc    :   
    多线程异步并发请求 vLLM inference -> batched denoise -> 并发仿真
    启动脚本 recogdrive/scripts/evaluation/inference_cot_negdrive.sh
'''

import sys

from typing import Any, Dict, List, Union, Tuple
from pathlib import Path
from dataclasses import asdict
from datetime import datetime
import traceback
import logging
import lzma
import pickle
import os
import uuid
from datetime import datetime

from concurrent.futures import ThreadPoolExecutor, as_completed
import requests
from tqdm import tqdm

from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

import torch
from torch.utils.data import DataLoader

import pickle
import io
import hydra
from hydra.utils import instantiate
from omegaconf import DictConfig
import pandas as pd

from nuplan.planning.script.builders.logging_builder import build_logger
from nuplan.planning.utils.multithreading.worker_utils import worker_map

from navsim.agents.negdrive.negdrive_backbone import NegDriveBackbone
from navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader
from navsim.common.dataclasses import SensorConfig
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.metric_caching.metric_cache import MetricCache




if __name__ == "__main__":
    
    vlm = NegDriveBackbone(model_type="internvl", checkpoint_path="/root/navsim_workspace/models/ReCogDrive-VLM-2B/")

    print(vlm.tokenizer.padding_side)