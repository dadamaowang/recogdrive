from navsim.common.dataloader import MetricCacheLoader

from pathlib import Path



if __name__ == "__main__":

    path = "/root/navsim_workspace/exps/metric_cache_train"

    metric_cache_loader = MetricCacheLoader(path)

    
    print(metric_cache_loader[1])