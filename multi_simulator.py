import argparse
import collections
import copy
import glob
import json
import math
import multiprocessing
import os
from xmlrpc.client import Boolean
import cvxpy as cp
import time

import numpy as np
import pandas
from random import shuffle

from cluster import Cluster
from pollux_mip import WeightedMIPPolicy
from gavel import GavelPolicy
from pollux_mip_fix import WeightedMIPFIXPolicy
from pmp import PmpPolicy
from utils import job_name_map


def multi_simulate(args):
    workload = pandas.read_csv(args.workload)

    # cluster_ngpus_per_node = {"aws" : 4, "dgx-ext" : 8, "rtx" : 8, "azure" : 8}
    # cluster_nnodes = {"aws" : 6, "dgx-ext" : 2, "rtx" : 3, "azure" : 5}
    # cluster_ngpus_per_node = {"aws" : 4, "dgx-ext" : 8, "rtx" : 8}
    # cluster_nnodes = {"aws" : 6, "dgx-ext" : 2, "rtx" : 3}
    # cluster_ngpus_per_node = {"rtx" : 8, "dgx-ext" : 8}
    # cluster_nnodes = {"rtx" : 3, "dgx-ext" : 2}
    cluster_ngpus_per_node = {"azure" : 8, "dgx-ext" : 8}
    cluster_nnodes = {"azure" : 5, "dgx-ext" : 2}

    if args.cluster_scale is not None:
        for k in cluster_nnodes.keys():
            cluster_nnodes[k] = cluster_nnodes[k] * args.cluster_scale
    cluster_max_physical_nnodes = {"aws" : 16, "rtx" : 3, "dgx-ext" : 2, "dgx" : 2, "azure" : 5}
    cluster_populate_nnodes = {k : min(v, cluster_max_physical_nnodes[k]) for k, v in cluster_nnodes.items()}
    
    timeshare_penalty_window = args.timeshare_sched_window if args.timeshare_sched_window > 0 else None

    if args.policy == "gavel":
        policy = GavelPolicy(args.interval, pmp=args.pmp)
        # policy = GavelPolicy(args.interval, policy='allox')
    elif args.policy == "weighted_mip_fix":
        policy = policy = WeightedMIPFIXPolicy(p_fairness=args.p, 
                                        restart_penalty=30,
                                        lambda_a=args.a,
                                        lambda_n=args.n,
                                        project_throughputs=args.project_throughputs,
                                        share_max_replicas=args.share_max_replicas,
                                        timeshare_penalty_window=timeshare_penalty_window)
    elif args.policy == "pmp":
        policy = PmpPolicy(args.interval, args.homo)
    else:
        policy = WeightedMIPPolicy(p_fairness=args.p, 
                                    restart_penalty=30,
                                    lambda_a=args.a,
                                    lambda_n=args.n,
                                    project_throughputs=args.project_throughputs,
                                    share_max_replicas=args.share_max_replicas,
                                    timeshare_penalty_window=timeshare_penalty_window,
                                    pmp=args.pmp,
                                    hete=args.hete)
    if isinstance(policy, WeightedMIPPolicy):
        policy.populate_valid_configs(cluster_populate_nnodes, cluster_ngpus_per_node)
        policy.populate_pmp_configs(cluster_nnodes, cluster_ngpus_per_node)
    elif isinstance(policy, GavelPolicy):
        policy.populate_valid_configs(cluster_nnodes, cluster_ngpus_per_node)
    cluster = Cluster(workload, policy, cluster_populate_nnodes, cluster_nnodes, cluster_ngpus_per_node, pmp = args.pmp)
    if args.oracle_num_nodes != 0:
        for job in cluster.jobs:
            job.seed_profiles(args.oracle_num_nodes, args.oracle_num_replicas)

    contain_pmp_jobs = False

    while not cluster.all_complete():
        # if args.debug and contain_pmp_jobs:
        if args.debug:
            input('\npress to continue')
        contain_pmp_jobs = False
        ####### STEP #######        
        cluster.step(args.interval)
        print("---------------- SIMULATOR TIME: {} ----------------"
              .format(cluster.current_time))
        if isinstance(policy, GavelPolicy):
            factor = 0 if len(cluster.load_factors) == 0 else cluster.load_factors[-1]
            print(f'Load factor: {factor}')
        print("Active jobs:")
        for val in cluster.logs[-1]["submitted_jobs"]:
            if val["submission_time"] <= cluster.current_time and val["completion_time"] is None:
                # print("    {}:\t\t[epoch {}]\t[restarts {}]\t[batch size {}]\t[placement {}]\t[n_avg {}]\t[cluster {}]".format(
                #       val["name"], val["epoch"], val["num_restarts"], val["batch_size"], val["placement"], val["n_avg"], val["cluster"]))
                print("    {}:\t\t[epoch {}]\t[restarts {}]\t[batch size {}]\t[placement {}]\t[n_avg {}]\t[overhead: {}]".format(
                      val["name"] if val["real_job_name"] is None else val["real_job_name"], val["epoch"], val["num_restarts"], val["batch_size"], val["placement"], val["n_avg"], val['overhead']))
                if val['name'].split('-')[0] in job_name_map:
                    contain_pmp_jobs = True
        # used_gpus = get_used_gpus(cluster.logs[-1], cluster.current_time)
        # print("GPU utilization: {}".format(used_gpus))
        print("Completed jobs:")
        jct_dict = cluster.get_jcts()
        print(jct_dict)
        print("Average JCT:", sum(jct_dict.values()) / len(jct_dict) if jct_dict else 0)
    
    # progress_track = cluster.get_progress_track()
    # print(progress_track)

    pmp_config_history = cluster.get_pmp_config_history()
    print('### pmp config history')
    for jobname, history in pmp_config_history.items():
        print(f'{jobname}: {history}')
    for job, history in pmp_config_history.items():
        pmp_config_history[job] = {str(k):v for k,v in history.items()}

    avg_load_factor = None
    if isinstance(policy, GavelPolicy):
        avg_load_factor = np.mean(cluster.load_factors)

    # num restarts
    num_restarts = cluster.get_num_restarts()
    
    if args.output:
        cluster.output_logs(args.output)
    return cluster.logs, cluster.get_jcts(), pmp_config_history, avg_load_factor, num_restarts

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
                        choices=["weighted_mip", "gavel", "weighted_mip_fix", "pmp"])
    parser.add_argument("--pmp", action='store_true', default=False, help="allow pmp")
    parser.add_argument("--p", type=float, default=0.5, help="value of p for policy=weighted_mip")
    parser.add_argument("--hete", action='store_true', default=False, help="use homogeneous allocations for pmp")
    parser.add_argument("--debug", action='store_true', default=False, help="press to continue at each round")
    parser.add_argument("--n", type=float, default=None, help="weighted_mip regularization: no-alloc")
    parser.add_argument("--a", type=float, default=None, help="weighted_mip regularization: change of alloc")
    parser.add_argument("--cluster_scale", type=int, default=None, help="scale of cluster relative to hardcoded values")
    parser.add_argument("--project_throughputs", action='store_true', default=False, help="projects throughput functions from one cluster to another by multiplying by a constant = ratio of mean of throughputs for num_replicas=1")
    parser.add_argument("--share_max_replicas", action='store_true', default=False, help="share max_profiled replicas across clusters")
    parser.add_argument("--oracle_num_nodes", type=int, default=0, help="max-num-nodes to seed profiles for, in each cluster")
    parser.add_argument("--oracle_num_replicas", type=int, default=0, help="number of replicas to seed profiles for")
    parser.add_argument("--timeshare_sched_window", type=int, default=0, help="timesharing window length")
    parser.add_argument("--interval", type=int, default=60,
                        help="scheduling interval in seconds")
    parser.add_argument("--output", type=str, help="path to output logs")

    
    args = parser.parse_args()
    if os.path.isdir(args.workload):
        print(args.output is not None, os.path.isdir(args.output))
        assert args.output is not None and os.path.isdir(args.output)
        args_list = []
        for workload in glob.glob(args.workload + "/*.csv"):
            name = os.path.basename(workload)[:-4]
            args_list.append(copy.deepcopy(args))
            args_list[-1].workload = workload
            args_list[-1].output = args.output + "/" + name + ".log"
        with multiprocessing.Pool(processes=32) as pool:
            ret_list = pool.map(multi_simulate, args_list)
        summary = {"jcts": {}, "avgs": {}, "avg_load_factor": {}}
        summary_configs = {}
        for args_item, (_, jct_dict, pmp_config_history, avg_load_factor, num_restarts) in zip(args_list, ret_list):
            name = os.path.basename(args_item.workload)[:-4]
            # summary["jcts"][name] = jct_dict
            # summary["num_restarts"][name] = num_restarts
            summary["jcts"][name] = {jobname: (jct_dict[jobname], num_restarts[jobname]) for jobname in jct_dict}
            summary["avgs"][name] = sum(jct_dict.values()) / len(jct_dict)
            summary["avg_load_factor"][name] = avg_load_factor
            summary_configs[name] = {str(k):v for k,v in pmp_config_history.items()}
        summary["mean"] = sum(summary["avgs"].values()) / len(summary["avgs"])
        with open(args.output + "/summary.json", "w") as f:
            json.dump(summary, f, indent=4)
        with open(args.output + "/summary_configs.json", "w") as f:
            json.dump(summary_configs, f, indent=4)
    else:
        multi_simulate(args)
