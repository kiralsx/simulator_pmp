import argparse
import collections
import copy
import glob
import json
import math
import multiprocessing
import os
import cvxpy as cp
import time

import numpy as np
from numpy.lib.function_base import select
import pandas
from random import shuffle

from scipy.linalg.misc import norm
from cluster_unaware import ClusterUnaware
from pollux_mip_unaware import WeightedMIPPolicyUnaware
from pollux_unaware import PolluxPolicyUnaware
from pollux_unaware_fix import PolluxPolicyUnawareFix

def multi_simulate(args):
    workload = pandas.read_csv(args.workload)

    cluster_ngpus_per_node = {"aws" : 4, "rtx" : 8, "dgx-ext" : 8}
    cluster_nnodes = {"aws" : 6, "rtx" : 3, "dgx-ext" : 2}
    if args.cluster_scale is not None:
        for k in cluster_nnodes.keys():
            cluster_nnodes[k] = cluster_nnodes[k] * args.cluster_scale
        print(f"Scaled up cluster_nnodes: {cluster_nnodes}")
    cluster_max_physical_nnodes = {"aws" : 16, "rtx" : 3, "dgx-ext" : 2, "dgx" : 2}
    cluster_populate_nnodes = {k : min(v, cluster_max_physical_nnodes[k]) for k, v in cluster_nnodes.items()}

    max_num_vnodes = max([cluster_populate_nnodes[c] * cluster_nnodes[c] for c in cluster_nnodes.keys()]) / min(cluster_ngpus_per_node.values())
    print(f"job max num replicas: {max_num_vnodes * min(cluster_populate_nnodes.values())}")

    if args.policy == "weighted_mip":
        policy = WeightedMIPPolicyUnaware(p_fairness=args.policy_p_val, 
                                        lambda_a=args.mip_lambda_a,
                                        lambda_n=args.mip_lambda_n)
        policy.populate_valid_configs(max_num_vnodes)
    elif args.policy == "pollux_fix":
        policy = PolluxPolicyUnawareFix(args.policy_p_val)
    else:
        policy = PolluxPolicyUnaware(args.policy_p_val)
    cluster = ClusterUnaware(workload, policy, cluster_nnodes, cluster_ngpus_per_node)

    current_time = 0
    while not cluster.all_complete():
        ####### STEP #######        
        cluster.step(args.interval)

        print("---------------- SIMULATOR TIME: {} ----------------"
              .format(cluster.current_time))
        print("Active jobs:")
        for val in cluster.logs[-1]["submitted_jobs"]:
            if val["submission_time"] <= cluster.current_time and val["completion_time"] is None:
                print("    {}:\t\t[epoch {}]\t[restarts {}]\t[batch size {}]\t[placement {}]\t[n_avg {}]\t[cluster {}]".format(
                      val["name"], val["epoch"], val["num_restarts"], val["batch_size"], val["placement"], val["n_avg"], val["cluster"]))
        used_gpus = get_used_gpus(cluster.logs[-1], cluster.current_time)
        print("GPU utilization: {}".format(used_gpus))
        print("Completed jobs:")
        jct_dict = cluster.get_jcts()
        print(jct_dict)
        print("Average JCT:", sum(jct_dict.values()) / len(jct_dict) if jct_dict else 0)
    
    if args.output:
        cluster.output_logs(args.output)
    return cluster.logs, cluster.get_jcts()

def get_used_gpus(log_entry, current_time):
    used_gpus = dict()
    for val in log_entry["submitted_jobs"]:
        if val["submission_time"] <= current_time and val["completion_time"] is None:
            cluster = val["cluster"]
            if cluster is None:
                continue
            else:
                if cluster not in used_gpus:
                    used_gpus[cluster] = 0
                used_gpus[cluster] += sum(val["placement"])
    return used_gpus

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("workload", type=str, help="path to workload csv")
    parser.add_argument("--policy", type=str, default="weighted_mip",
                        choices=["weighted_mip", "pollux", "pollux_fix"])
    parser.add_argument("--policy_p_val", type=float, default=0.5, help="value of p for policy=weighted_mip")
    parser.add_argument("--mip_lambda_n", type=float, default=None, help="weighted_mip regularization: no-alloc")
    parser.add_argument("--mip_lambda_a", type=float, default=None, help="weighted_mip regularization: change of alloc")
    parser.add_argument("--cluster_scale", type=int, default=None, help="scale of cluster relative to hardcoded config")
    parser.add_argument("--interval", type=int, default=60,
                        help="scheduling interval in seconds")
    parser.add_argument("--output", type=str,
                        help="path to output logs")
    
    args = parser.parse_args()
    if os.path.isdir(args.workload):
        assert args.output is not None and os.path.isdir(args.output)
        args_list = []
        for workload in glob.glob(args.workload + "/*.csv"):
            name = os.path.basename(workload)[:-4]
            args_list.append(copy.deepcopy(args))
            args_list[-1].workload = workload
            args_list[-1].output = args.output + "/" + name + ".log"
        with multiprocessing.Pool(processes=32) as pool:
            ret_list = pool.map(multi_simulate, args_list)
        summary = {"jcts": {}, "avgs": {}}
        for args_item, (_, jct_dict) in zip(args_list, ret_list):
            name = os.path.basename(args_item.workload)[:-4]
            summary["jcts"][name] = jct_dict
            summary["avgs"][name] = sum(jct_dict.values()) / len(jct_dict)
        summary["mean"] = sum(summary["avgs"].values()) / len(summary["avgs"])
        with open(args.output + "/summary.json", "w") as f:
            json.dump(summary, f, indent=4)
    else:
        multi_simulate(args)
