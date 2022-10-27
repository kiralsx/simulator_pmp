from pathlib import Path
import numpy as np
import json



iter_time_cache_path = f'./iteration_time.txt'
if Path(iter_time_cache_path).is_file():
    cache = json.load(open(iter_time_cache_path))
else:   
    cache = {}

job_name = 'gpt-1.3b'
num_layers = []
gpu_name = 'v100'
iter_time = {}
for config, time in cache[job_name].items():
    s = config.split('_')
    num_gpu_0 = int(s[0])
    gpu_0 = s[1]
    num_gpu_1 = int(s[2])
    gpu_1 = s[3]
    if gpu_0 == 'v100' and num_gpu_1 == 0:
        iter_time[num_gpu_0] = time
    
print(iter_time)


print({k: v/k for k,v in iter_time.items()})





job_list = ['gpt-1.3b', 'gpt-6.7b', 'gpt-15b', 'wresnet']





