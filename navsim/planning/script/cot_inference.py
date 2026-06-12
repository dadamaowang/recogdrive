# -*- encoding: utf-8 -*-
'''
@File    :   cot_inference.py
@Time    :   2026/06/11 16:28:40
@Author  :   Nuoqian Xiao
@Version :   0.0.1
@Contact :   feimaoxiaotianshi@outlook.com
@License :   (C)Copyright 2024-2025, Nuoqian Xiao
@Status  :   
@Desc    :   多线程异步并发请求 vLLM inference, 单 GPU 
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
                  image_b64, 
                  prompt,
                  api_base: str = "http://localhost/8001",
                  model_name : str = "InternVL", 
                  max_tokens : int = 128,
                  temperature: float = 0.0,     # TODO 评估通常使用 greedy decoding (0.0) 或较低温度
                  with_lora: bool = False,     # TODO with lora 


                  ):
    
    payload = {
        "model": model_name,  
        "messages": [
            {
                "role": "user",
                "content": [
                    # {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
                    {"type": "text", "text": "Hello, Who are you"}
                ]
            }
        ],
        "max_tokens": max_tokens,
        "temperature": temperature,  
        # "extra_body": {
        #     "lora_request": {
        #         "lora_name": lora_name,
        #         "lora_path": lora_path
        #     }
        # }
    }




    endpoints = ("/v1/generate", "/generate", "/v1/completions", "/completions", "/invoke", "/")




    try:
        for e in endpoints:
            url = api_base + (e if e.startswith("/") else ("/" + e))


            r = session.post(url, json=payload, timeout=120)

            print("成功地址")
            print(url)

            print(r)



    except Exception as e:
        print(e)


    
    
    # response = requests.post(f"{api_url}/v1/chat/completions", json=payload, timeout=120)
    # response.raise_for_status()
    # return response.json()['choices'][0]['message']['content']

    return






def inference_single_data_point(data_point, 
                                scene_loader,
                                metric_cache_loader,
                                simulator,
                                scorer, 
                                agent,
                                session,

                                
                                ):
    """Inference single scene
    
    


    """





    score_row: Dict[str, Any] = {"token": data_point, "valid": True}



    try:
        metric_cache_path = metric_cache_loader.metric_cache_paths[data_point]            
        with lzma.open(metric_cache_path, "rb") as f:
            metric_cache: MetricCache = pickle.load(f)

        requires_scene = False
        agent_input = scene_loader.get_agent_input_from_token(data_point)

        
        image64 = "im64"
        prompt = "p"
        call_vllm_api(session=session, image_b64=image64, prompt=prompt, )


        print("成功")

        # # concurrent requests
        # query = agent.prepare_for_vllm_service(agent_input)

        




            

        # trajectory = agent.compute_traj_cot(agent_input)
        # pdm_result = pdm_score(
        #     metric_cache=metric_cache,
        #     model_trajectory=trajectory,
        #     future_sampling=simulator.proposal_sampling,
        #     simulator=simulator,
        #     scorer=scorer,
        # )
        # score_row.update(asdict(pdm_result))


    except Exception as e:
        logger.warning(f"----------- Agent failed for token {data_point}:")
        traceback.print_exc()
        score_row["valid"] = False



    return score_row



def build_session(retries=10, backoff=0.5):
    s = requests.Session()
    retry = Retry(total=retries, backoff_factor=backoff,
                  status_forcelist=(429,500,502,503,504),
                  allowed_methods=frozenset(["GET"]))  # safer: avoid POST retries
    s.mount("https://", HTTPAdapter(max_retries=retry))
    s.mount("http://", HTTPAdapter(max_retries=retry))
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


    # TODO debug
    tokens_to_evaluate = tokens_to_evaluate[:35]

    final_results = []
    session = build_session()
    with ThreadPoolExecutor(max_workers=cfg.max_workers) as executor:
        futures = {
            executor.submit(
                inference_single_data_point,
                data_point,
                scene_loader, metric_cache_loader, simulator, scorer,
                agent,
                session
            ): data_point
            for data_point in tokens_to_evaluate
        }

        for future in tqdm(as_completed(futures), total=len(futures), desc="Evaluating Scenes"):
            try:
                result = future.result()
                final_results.append(result)
            except Exception:
                token = futures.get(future, "<unknown>")
                logger.warning(f"Future failed for token {token}:")
                traceback.print_exc()
    session.close()
    




    for d in final_results:
        print(d)


    
    print("SUCCESS")
    sys.exit(0)
    



    # 分发 data_points


# def run_pdm_score(args: List[Dict[str, Union[List[str], DictConfig]]]) -> List[Dict[str, Any]]:


#     pdm_results: List[Dict[str, Any]] = []
#     for idx, (token) in enumerate(tokens_to_evaluate):
#         if dist.get_rank() == 0:
#             logger.info(f"Rank {dist.get_rank()} processing scenario {idx+1} / {len(tokens_to_evaluate)} in thread_id={thread_id}, node_id={node_id}")

#         score_row: Dict[str, Any] = {"token": token, "valid": True}
#         try:
#             metric_cache_path = metric_cache_loader.metric_cache_paths[token]            
#             with lzma.open(metric_cache_path, "rb") as f:
#                 metric_cache: MetricCache = pickle.load(f)

#             requires_scene = False
#             agent_input = scene_loader.get_agent_input_from_token(token)
#             if requires_scene:
#                 raise NotImplementedError
#                 # scene = scene_loader.get_scene_from_token(token)
#                 # trajectory = agent.compute_trajectory(agent_input, scene)
#             else:
#                 """TODO customize 
                
#                 """
#                 # trajectory = agent.compute_trajectory_recogdrive(agent_input)
#                 trajectory = agent.compute_traj_cot(agent_input)
                
#             pdm_result = pdm_score(
#                 metric_cache=metric_cache,
#                 model_trajectory=trajectory,
#                 future_sampling=simulator.proposal_sampling,
#                 simulator=simulator,
#                 scorer=scorer,
#             )
#             score_row.update(asdict(pdm_result))
#             score_row['rank'] = dist.get_rank()
#         except Exception as e:
#             logger.warning(f"----------- Agent failed for token {token}:")
#             traceback.print_exc()
#             score_row["valid"] = False

#         pdm_results.append(score_row)
#     serialized_score_rows = pickle.dumps(pdm_results)
#     return serialized_score_rows





    # TODO: 并发请求 vLLM 服务



    # TODO: 整理数据


    serialized_score_rows = run_pdm_score(data_points)

    device = torch.device("cpu" if not torch.cuda.is_available() else "cuda")

    serialized_tensor = torch.ByteTensor(list(serialized_score_rows)).to(device)

    local_size = len(serialized_tensor)
    size_list = [torch.tensor(local_size).to(device) for _ in range(dist.get_world_size())]
    dist.all_gather(size_list, torch.tensor(local_size).to(device))

    max_size = max(size_list).item() 

    if local_size < max_size:
        padded_tensor = torch.cat([serialized_tensor, torch.zeros(max_size - local_size, dtype=torch.uint8).to(device)])
    else:
        padded_tensor = serialized_tensor

    gathered_results = [torch.empty_like(padded_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_results, padded_tensor)

    if local_size < max_size:
        padded_tensor = torch.cat([serialized_tensor, torch.zeros(max_size - local_size, dtype=torch.uint8).to(device)])
    else:
        padded_tensor = serialized_tensor

    gathered_results = [torch.empty_like(padded_tensor) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_results, padded_tensor)

    if dist.get_rank() == 0:
        final_results = []
        for gathered_tensor in gathered_results:
            gathered_tensor = gathered_tensor[:local_size]  
            serialized_data = gathered_tensor.cpu().numpy().tobytes()
            final_results.extend(pickle.loads(serialized_data))  # 
    
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