import argparse
import collections
import copy
import glob
import math
import json
import multiprocessing
import os
import csv
import time
from subprocess import check_output

import numpy as np
import pandas

from applications import APPLICATIONS
from pollux_mip import WeightedMIPPolicy
from pollux_unaware import PolluxPolicyUnaware
from utils import JobInfo, JobInfoUnaware, NodeInfo
from gavel import GavelPolicy
from job import Job

class FTFCluster(object):
    # num_nodes, ngpus_per_node are dicts with key=cluster_name
    def __init__(self, workload, policy, num_nodes, ngpus_per_node, job_max_replicas):
        self.workload = workload
        self.policy = policy
        self.num_nodes = num_nodes
        self.num_gpus = ngpus_per_node
        self.current_time = 0
        self.clusters = list(self.num_nodes.keys())
        self.job_max_replicas = job_max_replicas
        
        # create inverse map like {app-name: {cluster_name: cluster_app}}
        self.cluster_applications = dict()
        for cluster_name, cluster_apps in APPLICATIONS.items():
            for app_name, app in cluster_apps.items():
                if app_name not in self.cluster_applications:
                    self.cluster_applications[app_name] = dict()
                self.cluster_applications[app_name][cluster_name] = app
        
        if isinstance(policy, WeightedMIPPolicy) or isinstance(policy, PolluxPolicyUnaware):
            cache_speedups = isinstance(policy, PolluxPolicyUnaware)
            self.jobs = [Job(row.name, self.cluster_applications[row.application], 
                             row.time, cache_speedups=cache_speedups) for row in workload.itertuples()]
            cname = next(iter(self.num_nodes.keys()))
            for job in self.jobs:
                if job.applications[cname].name == "ncf":
                    job.target_batch_size = 32768
        elif isinstance(policy, GavelPolicy):
            # change the target num_replicas to fit into the smallest cluster, e.g. imagenet 32 -> 16
            min_cluster_size = min([self.num_nodes[k] * self.num_gpus[k] for k in self.num_nodes.keys()])
            self.jobs = [Job(row.name, self.cluster_applications[row.application], row.time,
                             target_num_replicas=min(row.num_replicas, min_cluster_size),
                             target_batch_size=row.batch_size)
                         for row in workload.itertuples()]
            for job in self.jobs:
                job.fix_minibatch_size()
        
        self.allocations = {}
        self.logs = []
        self.utility = []

    def step(self, seconds=60):
        self.step_jobs(seconds)
        self.optimize_policy()
        self.step_log()

    def step_jobs(self, seconds=60):
        check_interference = False
        # TODO: fix interference check code
        if check_interference:
            interfere_nodes = set(idx for idx in range(self.num_nodes) 
                              if sum(len(set(val)) > 1 and idx in val
                                     for key, val in self.allocations.items()) > 1)
        for job in self.jobs:
            if check_interference:
                alloc_set = set(self.allocations.get(job.name, [])[1])
                interference = 0.0 if len(alloc_set) > 1 and any(idx in interfere_nodes for idx in alloc_set) else 0.0
            else:
                interference = 0.0
            job.step(seconds, interference=interference)
            
        self.current_time += seconds
    
    def optimize_policy(self):
        job_infos = self.get_job_infos()
        if job_infos:
            # Optimize allocations.
            node_infos = self.get_node_infos()
            self.allocations = {k: v for k, v in self.allocations.items() if k in job_infos}
            
            cluster = self.clusters[0]
            # run scheduling policy
            p_start_t = time.time()
            if isinstance(self.policy, PolluxPolicyUnaware):
                allocations, _ = self.policy.optimize(job_infos, node_infos[cluster], self.allocations, node_infos[cluster][0])
            elif isinstance(self.policy, GavelPolicy):
                allocations = self.policy.optimize(job_infos, node_infos, self.allocations)
            else:
                allocations = self.policy.optimize(job_infos, node_infos, self.allocations)
            p_end_t = time.time()
            # print(f"policy.optimize(): {round((p_end_t-p_start_t)*1000)}ms")
            
            for job in self.jobs:
                if job.submission_time <= self.current_time and job.completion_time is None:
                    job.contention.append(len(job_infos))
                    # print(f"job: {job.name}, contention: {job.contention}")
                    if isinstance(self.policy, WeightedMIPPolicy) or isinstance(self.policy, GavelPolicy):
                        cluster, alloc = allocations.get(job.name, (None, ()))
                        # change in resources
                        if allocations.get(job.name) != self.allocations.get(job.name):
                            placement = []
                            for i in range(len(alloc)):
                                if i == 0 or alloc[i] != alloc[i - 1]:
                                    placement.append(1)
                                else:
                                    placement[-1] += 1
                            if job.current_cluster != cluster:
                                # migrated to another cluster
                                job.migrate(cluster, placement)
                            elif self.allocations.get(job.name) != alloc:
                                # reallocated within same cluster
                                job.reallocate(placement)
                    elif isinstance(self.policy, PolluxPolicyUnaware):
                        alloc = allocations.get(job.name, ())
                        # change in resources
                        if allocations.get(job.name) != self.allocations.get(job.name):
                            placement = []
                            for i in range(len(alloc)):
                                if i == 0 or alloc[i] != alloc[i - 1]:
                                    placement.append(1)
                                else:
                                    placement[-1] += 1
                            if job.current_cluster != cluster:
                                # migrated to another cluster
                                job.migrate(cluster, placement)
                            elif self.allocations.get(job.name) != alloc:
                                # reallocated within same cluster
                                job.reallocate(placement)
                    else:
                        assert False, "other policies not implemented"
            # make a copy of allocations
            self.allocations = allocations

    def step_log(self):
        step_log = {
            "timestamp": self.current_time,
            "num_nodes": self.num_nodes,
            "submitted_jobs": [
                {
                    "name": job.name,
                    "epoch": job.epoch,
                    "progress": job.progress,
                    "num_restarts": job.num_restarts,
                    "cluster": job.current_cluster,
                    "allocation": self.allocations.get(job.name, tuple()),
                    "placement": job.placement,
                    "batch_size": job.atomic_bsz * (job.accum_steps + 1) * int(sum(job.placement)),
                    "accum_steps": job.accum_steps,
                    "submission_time": job.submission_time,
                    "start_time": job.start_time,
                    "completion_time": job.completion_time,
                    "n_avg" : np.mean(np.asarray(job.contention))
                }
                for job in self.jobs if job.submission_time <= self.current_time
            ],
        }
        self.logs.append(step_log)

    def get_job_infos(self):
        job_infos = {}
        for job in self.jobs:
            if self.current_time >= job.submission_time and job.completion_time is None:
                if isinstance(self.policy, WeightedMIPPolicy):
                    job_info = self.get_weighted_mip_multi_job_info(job)
                elif isinstance(self.policy, PolluxPolicyUnaware):
                    job_info = self.get_pollux_job_info(job)
                elif isinstance(self.policy, GavelPolicy):
                    job_info = self.get_gavel_job_info(job)
                else:
                    print(f"Unknown policy")
                job_infos[job.name] = job_info
        return job_infos
    
    def get_pollux_job_info(self, job):
        cluster = self.clusters[0]
        speedup_fn = job.get_speedup_fn(cluster)
        max_replicas = min(max(2 * job.max_profiled_replicas(cluster), 1), 
                           self.job_max_replicas,  # simulator can't handle more.
                           job.applications[cluster].max_batch_size // job.applications[cluster].min_local_bsz)
        job_info = JobInfoUnaware(
            resources={"nvidia.com/gpu": 1},
            speedup_fn=speedup_fn,
            creation_timestamp=job.submission_time,
            attained_service=job.attained_service,
            min_replicas= 0,
            max_replicas= max_replicas,
            preemptible=True,
        )
        
        app = job.applications[cluster]
        if app.name == "ncf":
            job_info.max_replicas = 1
        job_info.num_restarts = job.num_restarts or 0
        job_info.num_migrations = job.num_migrations or 0
        job_info.age = self.current_time - job.submission_time
        return job_info

    def get_gavel_job_info(self, job):
        speedup_fns = {cname : job.get_speedup_fn(cname) for cname in self.clusters}
        max_replicas = {cname : min(max(2 * job.max_profiled_replicas(cname), 1), 64,  # simulator can't handle more.
                        job.applications[cname].max_batch_size // job.applications[cname].min_local_bsz)
                        for cname in self.clusters}
        job_info = JobInfo(
            resources={"nvidia.com/gpu": 1},
            speedup_fn=speedup_fns,
            creation_timestamp=job.submission_time,
            attained_service=job.attained_service,
            min_replicas= {cname : 0 for cname in self.clusters},
            #max_replicas=min(max(2 * job.max_profiled_replicas, 1), 64,  # simulator can't handle more.
            #                 job.target_batch_size // job.application.min_local_bsz),
            max_replicas=max_replicas,
            preemptible=True,
        )
        for cname, capp in job.applications.items():
            if capp.name == "ncf":
                job_info.max_replicas[cname] = 1
        job_info.applications = job.applications
        job_info.epoch = job.epoch
        job_info.target_batch_size = job.target_batch_size
        job_info.scale_factor = job.target_num_replicas
        job_info.age = self.current_time - job.submission_time
        return job_info

    def get_weighted_mip_multi_job_info(self, job):
        speedup_fns = {cname : job.get_speedup_fn(cname) for cname in self.clusters}
        max_replicas = {cname : min(max(2 * job.max_profiled_replicas(cname), 1), self.job_max_replicas,  # simulator can't handle more.
                        job.applications[cname].max_batch_size // job.applications[cname].min_local_bsz)
                        for cname in self.clusters}
        job_info = JobInfo(
            resources={"nvidia.com/gpu": 1},
            speedup_fn=speedup_fns,
            creation_timestamp=job.submission_time,
            attained_service=job.attained_service,
            min_replicas= {cname : 0 for cname in self.clusters},
            max_replicas= max_replicas,
            preemptible=True,
        )
        
        for cname, capp in job.applications.items():
            if capp.name == "ncf":
                job_info.max_replicas[cname] = 1
        job_info.num_restarts = job.num_restarts or 0
        job_info.num_migrations = job.num_migrations or 0
        job_info.age = self.current_time - job.submission_time
        return job_info

    def get_node_infos(self, num_nodes=None):
        cluster_node_info = dict()
        for cluster_name in self.clusters:
            cluster_info = {idx: NodeInfo({"nvidia.com/gpu": self.num_gpus[cluster_name]}, preemptible=False)
                            for idx in range(self.num_nodes[cluster_name])}
            cluster_node_info[cluster_name] = cluster_info
        return cluster_node_info

    def all_complete(self):
        return all(job.completion_time is not None for job in self.jobs)

    def output_logs(self, path):
        with open(path, "w") as f:
            for record in self.logs:
                json.dump(record, f)
                f.write("\n")

    def get_jcts(self):
        if len(self.logs) > 0:
            return {
                val["name"]: val["completion_time"] - val["submission_time"]
                for val in self.logs[-1]["submitted_jobs"]
                if val["completion_time"] is not None 
               }
        else:
            return {}

def simulate_single_job(args, job_df, max_num_replicas, cluster_name, cluster_nnodes, cluster_ngpus_per_node):
    # early exit for no run
    if max_num_replicas == 0:
        return 0

    # instantiate simulator
    job_max_num_replicas = max_num_replicas
    assert job_max_num_replicas > 0, "invalid job num replicas"
    job_max_nodes = int(job_max_num_replicas // cluster_ngpus_per_node[cluster_name]) + 1

    # job scales to cluster size at most
    job_max_nodes = min(cluster_nnodes[cluster_name], job_max_nodes)
    job_max_num_replicas = min(job_max_nodes * cluster_ngpus_per_node[cluster_name], job_max_num_replicas)

    # keep submission time relative to scheduling interval same
    job_nnodes = {cluster_name : job_max_nodes}
    job_ngpus_per_node = {cluster_name : cluster_ngpus_per_node[cluster_name]}

    if args.policy == "weighted_mip":
        policy = WeightedMIPPolicy(p_fairness=args.policy_p_val, 
                                   lambda_n=args.mip_lambda_n,
                                   lambda_a=args.mip_lambda_a)
        policy.populate_valid_configs(job_nnodes, job_ngpus_per_node)
    elif args.policy == "pollux":
        policy = PolluxPolicyUnaware(p_fairness=args.policy_p_val)
    elif args.policy == "gavel":
        policy = GavelPolicy(args.interval)
        policy.populate_valid_configs(job_nnodes, job_ngpus_per_node)

    job_df.time = (job_df.time % args.interval)
    simulator = FTFCluster(job_df, policy, job_max_replicas=job_max_num_replicas,
                           num_nodes = job_nnodes, ngpus_per_node = job_ngpus_per_node)


    while not simulator.all_complete():
        print(f"{simulator.jobs[0].name}, num_replicas: {job_max_num_replicas}, cluster_size: {job_max_nodes}, cluster: {cluster_name}")
        st_time = time.time()
        simulator.step(args.interval)
        end_time = time.time()
    return simulator.get_jcts()

def generate_workload_jobs(workload_file, args, cluster_nnodes, cluster_ngpus_per_node):
    # read workload
    workload = pandas.read_csv(workload_file)
    
    # read logs from previous run
    workload_name = os.path.basename(workload_file)
    logfile = os.path.join(args.logdir, workload_name.replace(".csv", ".log"))
    out = check_output(["tail", "-n1", logfile])
    last_log = json.loads(out)
    num_replicas = {}
    total_num_gpus = sum([cluster_nnodes[k] * cluster_ngpus_per_node[k] for k in cluster_ngpus_per_node.keys()])

    print(f"Isolated cluster sizes:")
    for job in last_log['submitted_jobs']:
        num_replicas[job['name']] = total_num_gpus * 1.0 / job['n_avg']
        print(f"{job['name']} : {num_replicas[job['name']]}")
    
    lower_max_replicas, upper_max_replicas = {}, {}
    args_map = {}
    for k, v in num_replicas.items():
        if v < 1:
            print(f"WARNING:: {k} has n_avg: {last_log['submitted_jobs'][k]['n_avg']}, isolated_cluster_size: {v}")
            lower_max_replicas[k] = 0
            upper_max_replicas[k] = 1
        else:
            lower_max_replicas[k] = max(math.floor(v), 1)
            upper_max_replicas[k] = max(math.ceil(v), 1)
        cluster_args = {}
        for cluster_name in cluster_nnodes.keys():
            lower_args = (args, workload[workload.name == k], lower_max_replicas[k], 
                          cluster_name, cluster_nnodes, cluster_ngpus_per_node)
            upper_args = (args, workload[workload.name == k], upper_max_replicas[k], 
                          cluster_name, cluster_nnodes, cluster_ngpus_per_node)
            cluster_args[cluster_name] = (lower_args, upper_args, v)
        args_map[k] = cluster_args

    return args_map

# use a weighted-mean where weights are total-num-gpus in each cluster
def combine_cluster_jcts(cluster_jcts, cluster_nnodes, cluster_ngpus_per_node):
  jct = 0
  weight_sum = 0
  for cluster_name in cluster_nnodes.keys():
    weight = cluster_ngpus_per_node[cluster_name] * cluster_nnodes[cluster_name]
    jct += weight * cluster_jcts[cluster_name]
    weight_sum += weight
  return jct / weight_sum

def simulate_dir(args, cluster_nnodes, cluster_ngpus_per_node):
    args_dict = {}
    dict_to_arglist_idxmap = {}
    args_list = []
    idx = 0
    for workload_file in glob.glob(args.workload + "/*.csv"):
      workload_name = os.path.basename(workload_file).replace(".csv", "")
      args_dict[workload_name] = generate_workload_jobs(workload_file, args, cluster_nnodes, cluster_ngpus_per_node)
      dict_to_arglist_idxmap[workload_name] = {}
      for jobname, cluster_args in args_dict[workload_name].items():
        dict_to_arglist_idxmap[workload_name][jobname] = dict()
        for cluster_name, cluster_arg in cluster_args.items():
            lower_cluster_arg, upper_cluster_arg, _ = cluster_arg
            args_list.append(lower_cluster_arg)
            args_list.append(upper_cluster_arg)
            dict_to_arglist_idxmap[workload_name][jobname][cluster_name] = (idx, idx + 1)
            idx += 2

    jcts_list = []
    '''
    for args in args_list:
        jct_dict = simulate_single_job(*args)
        jcts_list.append(jct_dict)
    '''
    with multiprocessing.Pool(processes=args.num_procs) as pool:
      jcts_list = pool.starmap(simulate_single_job, args_list)

    def interpolate_jct(jobname, lower_idx, upper_idx, num_replicas):
        lower_jct_dict, upper_jct_dict = jcts_list[lower_idx], jcts_list[upper_idx]
        lower_jct, upper_jct = lower_jct_dict[jobname], upper_jct_dict[jobname]
        if num_replicas < 1:
            jct = lower_jct / num_replicas
        else:
            lower_max_replicas, upper_max_replicas = max(math.floor(num_replicas), 1), max(math.ceil(num_replicas), 1)
            jct = lower_jct
            
            # if needs interpolation
            if upper_max_replicas > lower_max_replicas:
                x_delta = (num_replicas - lower_max_replicas)
                jct += (upper_jct - lower_jct) * x_delta
        return jct
    
    # interpolate jct
    summary_dict = {"jcts" : {}}
    for workload_name in dict_to_arglist_idxmap.keys():
      workload_jct_dict = {}
      for jobname, job_cluster_idxs_dict in dict_to_arglist_idxmap[workload_name].items():
        cluster_jcts = dict()
        for cluster_name, cluster_idxs in job_cluster_idxs_dict.items():
            lower_idx, upper_idx = cluster_idxs
            _, _, num_replicas = args_dict[workload_name][jobname][cluster_name]
            cluster_jct = interpolate_jct(jobname=jobname, lower_idx=lower_idx, 
                                            upper_idx=upper_idx, num_replicas=num_replicas)
            cluster_jcts[cluster_name] =  cluster_jct
        print(f"Job{jobname}: JCTs: {cluster_jcts}")
        workload_jct_dict[jobname] = combine_cluster_jcts(cluster_jcts, cluster_nnodes, cluster_ngpus_per_node)
        summary_dict["jcts"][workload_name] = workload_jct_dict

    # compute statistics
    avg_jct_dict = {}
    for workload, workload_jct_dict in summary_dict["jcts"].items():
      avg_jct_dict[workload] = sum(workload_jct_dict.values()) / len(workload_jct_dict)
    summary_dict["avgs"] = avg_jct_dict
    summary_dict["mean"] = sum(avg_jct_dict.values()) / len(avg_jct_dict)
    
    return summary_dict

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("workload", type=str, help="path to workload csv")
  parser.add_argument("--logdir", type=str, help="path to workload logs from previous run")
  parser.add_argument("--policy", type=str, default="pollux",
                      choices=["pollux", "weighted_mip", "gavel"])
  parser.add_argument("--policy_p_val", type=float, default=-1, help="value of p for policy=[pollux/weighted_mip]")
  parser.add_argument("--mip_lambda_n", type=float, default=None, help="weighted_mip regularization: no-alloc")
  parser.add_argument("--mip_lambda_a", type=float, default=None, help="weighted_mip regularization: change of alloc")
  parser.add_argument("--interval", type=int, default=60, help="scheduling interval in seconds")
  parser.add_argument("--num_procs", type=int, default=16, help="number of parallel processes to run FTF eval using")
  parser.add_argument("--output", type=str, help="path to output logs")
  # define cluster config
  cluster_nnodes = {"aws" : 6, "rtx" : 3, "dgx-ext" : 2}
  cluster_ngpus_per_node = {"aws" : 4, "rtx" : 8, "dgx-ext" : 8}
  args = parser.parse_args()
  if not os.path.isdir(args.workload):
    print("can only run on entire workload dir")
  else:
    summary = simulate_dir(args, cluster_nnodes, cluster_ngpus_per_node)
    with open(args.output + "/ftf_summary.json", "w") as f:
      json.dump(summary, f, indent=4)
