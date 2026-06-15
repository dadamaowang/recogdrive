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

from navsim.agents.abstract_agent import AbstractAgent
from navsim.common.dataloader import SceneLoader, SceneFilter, MetricCacheLoader
from navsim.common.dataclasses import SensorConfig
from navsim.evaluate.pdm_score import pdm_score
from navsim.planning.script.builders.worker_pool_builder import build_worker
from navsim.planning.simulation.planner.pdm_planner.simulation.pdm_simulator import PDMSimulator
from navsim.planning.simulation.planner.pdm_planner.scoring.pdm_scorer import PDMScorer
from navsim.planning.metric_caching.metric_cache import MetricCache

logger = logging.getLogger(__name__)

CONFIG_PATH = "config/pdm_scoring"
CONFIG_NAME = "default_run_pdm_score"


def call_vllm_api(session,
                  vllm_input_messages,
                  api_base: str = "http://localhost:8001",
                  model_name : str = "InternVL", 
                  max_tokens : int = 128,
                  temperature: float = 0.0,     # TODO 评估通常使用 greedy decoding (0.0) 或较低温度
                  with_lora: bool = False,     # TODO with lora 
                  ):
    """
    
    vllm_input_messages: 
    [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                        {"type": "text", "text": "Hello, Who are you"}
                    ]
                }
    ],
    
    """
    url = f"{api_base}/v1/chat/completions"  
    
    payload = {
        "model": model_name,  
        "messages": vllm_input_messages,
        "max_tokens": max_tokens,
        "temperature": temperature,  
        # "extra_body": {   # TODO add Lora
        #     "lora_request": {
        #         "lora_name": lora_name,
        #         "lora_path": lora_path
        #     }
        # }
    }

    try:
        r = session.post(url, json=payload, timeout=120)
        r.raise_for_status()

        result = r.json()
        text = result["choices"][0]["message"]["content"]

        print(f'CoT Response：{text}')
        return text

    except Exception as e:
        print("vLLM call failed:", e)
        if 'r' in locals():
            print("Response:", r.text)
        return None


def inference_single_data_point(data_point, 
                                scene_loader,
                                agent,
                                session,
                                cfg,
                                ):
    """Inference single scene
    
    """
    score_row: Dict[str, Any] = {"token": data_point, "valid": True}

    try:
        agent_input = scene_loader.get_agent_input_from_token(data_point)

        vllm_input_messages = agent.unpack_features_for_vllm_service(agent_input)
        if vllm_input_messages is None:
            score_row["valid"] = False
            return score_row

        cot_text = call_vllm_api(
                             session=session, 
                             vllm_input_messages=vllm_input_messages,
                             api_base=cfg.api_base,
                             model_name=agent.vlm_path,
                             max_tokens=agent.max_text_tokens,
                             temperature=cfg.temperature
                             )
        score_row["cot"] = cot_text if cot_text is not None else ""
        
    except Exception as e:
        logger.warning(f"----------- Agent failed for token {data_point}:")
        traceback.print_exc()
        score_row["valid"] = False

    return score_row



def build_session(retries=3, backoff=0.5):
    s = requests.Session()
    retry = Retry(total=retries, backoff_factor=backoff,
                  status_forcelist=(429,500,502,503,504),
                  allowed_methods=frozenset(["GET"]))  # safer: avoid POST retries
    s.mount("https://", HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=100))
    s.mount("http://", HTTPAdapter(max_retries=retry, pool_connections=10, pool_maxsize=100))
    return s



@hydra.main(config_path=CONFIG_PATH, config_name=CONFIG_NAME, version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Main entrypoint for running PDMS evaluation.
    :param cfg: omegaconf dictionary
    """

    build_logger(cfg)

    # =====================================
    # Prepare test data, simulator, agent
    # =====================================
    simulator: PDMSimulator = instantiate(cfg.simulator)
    scorer: PDMScorer = instantiate(cfg.scorer)
    assert (
        simulator.proposal_sampling == scorer.proposal_sampling
    ), "Simulator and scorer proposal sampling has to be identical"

    metric_cache_loader = MetricCacheLoader(Path(cfg.metric_cache_path))

    scene_loader = SceneLoader(
        sensor_blobs_path=None,
        data_path=Path(cfg.navsim_log_path),
        scene_filter=instantiate(cfg.train_test_split.scene_filter),
        sensor_config=SensorConfig.build_no_sensors(),
    )
    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    tokens_to_evaluate = sorted(tokens_to_evaluate)  
    num_missing_metric_cache_tokens = len(set(scene_loader.tokens) - set(metric_cache_loader.tokens))
    num_unused_metric_cache_tokens = len(set(metric_cache_loader.tokens) - set(scene_loader.tokens))
    if num_missing_metric_cache_tokens > 0:
        logger.warning(f"Missing metric cache for {num_missing_metric_cache_tokens} tokens. Skipping these tokens.")
    if num_unused_metric_cache_tokens > 0:
        logger.warning(f"Unused metric cache for {num_unused_metric_cache_tokens} tokens. Skipping these tokens.")
    
    logger.info("Starting pdm scoring of %s scenarios...", str(len(tokens_to_evaluate)))


    data_points = []
    for token in tokens_to_evaluate:
        log_file = scene_loader.token_to_log_file[token] 
        data_points.append({
            "cfg": cfg,
            "log_file": log_file,
            "tokens": [token],
        })
    log_names = [a["log_file"] for a in data_points]
    tokens = [t for a in data_points for t in a["tokens"]]

    agent: AbstractAgent = instantiate(cfg.agent)

    scene_filter: SceneFilter = instantiate(cfg.train_test_split.scene_filter)
    scene_filter.log_names = log_names
    scene_filter.tokens = tokens

    scene_loader = SceneLoader(
        sensor_blobs_path=Path(cfg.sensor_blobs_path),
        data_path=Path(cfg.navsim_log_path),
        scene_filter=scene_filter,
        sensor_config=agent.get_sensor_config(),
        load_image_path=True
    )
    tokens_to_evaluate = list(set(scene_loader.tokens) & set(metric_cache_loader.tokens))
    tokens_to_evaluate = sorted(tokens_to_evaluate) 

    print("TOKENS TO EVALUATE: %s", str(len(tokens_to_evaluate)))


    # =====================================
    #   Call vLLM Inference
    # =====================================

    # TODO 
    tokens_to_evaluate = tokens_to_evaluate[:10]

    final_results = []
    session = build_session()
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
        futures = {
            executor.submit(
                inference_single_data_point,
                data_point,
                scene_loader,
                agent,
                session,
                cfg

            ): data_point
            for data_point in tokens_to_evaluate
        }
        for future in tqdm(as_completed(futures), total=len(futures), desc="vLLM Inference"):
            try:
                result = future.result()
                final_results.append(result)
            except Exception:
                token = futures.get(future, "<unknown>")
                logger.warning(f"Future failed for token {token}:")
                traceback.print_exc()
    session.close()

    import pickle
    size_bytes = len(pickle.dumps(final_results))
    print(f"final_results pickle size: {size_bytes} bytes ({size_bytes/1024**2:.2f} MB)")
        
    # =====================================
    #   Batched Score Inference
    # =====================================
    for i in tqdm(range(0, len(final_results)), desc="Denoising and Evaluate"):     

        agent_input = scene_loader.get_agent_input_from_token(final_results[i]["token"])
        trajectory = agent.compute_traj_cot(agent_input, final_results[i]["cot"])

        metric_cache_path = metric_cache_loader.metric_cache_paths[final_results[i]["token"]]            
        with lzma.open(metric_cache_path, "rb") as f:
            metric_cache: MetricCache = pickle.load(f)

        pdm_result = pdm_score(
            metric_cache=metric_cache,
            model_trajectory=trajectory,
            future_sampling=simulator.proposal_sampling,
            simulator=simulator,
            scorer=scorer,
        )
        final_results[i].update(asdict(pdm_result))
    
    # =====================================
    #  Summary Final Result
    # =====================================    

    pdm_score_df = pd.DataFrame(final_results)

    num_sucessful_scenarios = pdm_score_df["valid"].sum()
    num_failed_scenarios = len(pdm_score_df) - num_sucessful_scenarios
    average_row = pdm_score_df.drop(columns=["token", "valid",'rank']).mean(skipna=True)
    average_row["token"] = "average"
    average_row["valid"] = pdm_score_df["valid"].all()
    average_row["rank"] = "0"
    pdm_score_df.loc[len(pdm_score_df)] = average_row

    save_path = Path(cfg.output_dir)
    timestamp = datetime.now().strftime("%Y.%m.%d.%H.%M.%S")
    pdm_score_df.to_csv(save_path / f"{timestamp}.csv")

    logger.info(
        f"""
        Finished running evaluation.
            Number of successful scenarios: {num_sucessful_scenarios}.
            Number of failed scenarios: {num_failed_scenarios}.
            Final average score of valid results: {pdm_score_df['score'].mean()}.
            Results are stored in: {save_path / f"{timestamp}.csv"}.
        """
    )


if __name__ == "__main__":
    main()