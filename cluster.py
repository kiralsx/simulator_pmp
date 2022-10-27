import collections
import copy
import json
import csv
import time
import numpy as np

from applications import APPLICATIONS
from pollux_mip import WeightedMIPPolicy, slice_cluster
from utils import JobInfo, NodeInfo, job_name_map, order_cluster
from gavel import GavelPolicy
from job import Job
from pollux_mip_fix import WeightedMIPFIXPolicy
from pmp import PmpPolicy
import random

class Cluster(object):
    # num_nodes, ngpus_per_node are dicts with key=cluster_name
    def __init__(self, workload, policy, cluster_populate_nnodes, num_nodes, ngpus_per_node, pmp=False):
        self.workload = workload
        self.policy = policy
        self.cluster_populate_nnodes = cluster_populate_nnodes
        self.num_nodes = num_nodes
        self.num_gpus = ngpus_per_node
        self.current_time = 0
        self.clusters = order_cluster(list(self.num_nodes.keys()))
        self.allow_pmp = pmp

        self.load_factors = []
        force_one_gpu = False
        
        # create inverse map like {app-name: {cluster_name: cluster_app}}
        self.cluster_applications = dict()
        for cluster_name, cluster_apps in APPLICATIONS.items():
            for app_name, app in cluster_apps.items():
                if app_name not in self.cluster_applications:
                    self.cluster_applications[app_name] = dict()
                self.cluster_applications[app_name][cluster_name] = app

        
        if isinstance(policy, WeightedMIPPolicy):
            # cache_speedups = isinstance(policy, PolluxPolicy)
            cache_speedups = False
            self.jobs = [Job(row.name, self.cluster_applications[row.application], 
                             row.time, cache_speedups=cache_speedups, app_name=row.application, target_batch_size=row.batch_size, allow_pmp=self.allow_pmp) for row in workload.itertuples()]
            # NOTE: set target batch size for pmp jobs only
            seed = 0
            random.seed(seed)
            for job in self.jobs:
                if not self.allow_pmp or job.app_name not in job_name_map:
                    job.target_batch_size = None   
                # NOTE: don't have stats efficiency for bsz > 364 for bert         

        elif isinstance(policy, GavelPolicy):
            # change the target num_replicas to fit into the smallest cluster, e.g. imagenet 32 -> 16
            min_cluster_size = min([self.cluster_populate_nnodes[k] * self.num_gpus[k] for k in self.cluster_populate_nnodes.keys()])

            if force_one_gpu:
                raise not NotImplementedError
                self.jobs = [Job(row.name, self.cluster_applications[row.application], row.time,
                            target_num_replicas=1,
                            target_batch_size=row.batch_size)
                        for row in workload.itertuples()]
            else:
                self.jobs = [Job(row.name, self.cluster_applications[row.application], row.time,
                                app_name=row.application,
                                target_num_replicas=min(row.num_replicas, min_cluster_size),
                                target_batch_size=row.batch_size,
                                allow_pmp=self.allow_pmp)
                            for row in workload.itertuples()]
            # fix bsz for gavel
            for job in self.jobs:
                job.fix_minibatch_size()
            cname = next(iter(self.num_nodes.keys()))
        elif isinstance(policy, WeightedMIPFIXPolicy):
            # cache_speedups = isinstance(policy, PolluxPolicy)
            cache_speedups = False
            min_cluster_size = min([self.num_nodes[k] * self.num_gpus[k] for k in self.num_nodes.keys()])
            self.jobs = [Job(row.name, self.cluster_applications[row.application], 
                             row.time, cache_speedups=cache_speedups,
                             target_num_replicas=min(row.num_replicas, min_cluster_size) if not force_one_gpu else 1,
                             target_batch_size=row.batch_size) 
                        for row in workload.itertuples()]
        # elif isinstance(policy, PmpPolicy) :
        #     # cache_speedups = isinstance(policy, PolluxPolicy)
        #     cache_speedups = False
        #     self.jobs = [Job(row.name, self.cluster_applications[row.application], 
        #                      row.time, cache_speedups=cache_speedups,
        #                      target_batch_size=row.batch_size) 
        #                 for row in workload.itertuples()]
        #     cname = next(iter(self.num_nodes.keys()))
        #     for job in self.jobs:
        #         if job.applications[cname].name == "ncf":
        #             job.target_batch_size = 32768
        else:
            assert False, f"unsupported policy {policy.__class__.__name__}"
        self.allocations = {}
        self.logs = []
        self.utility = []
        self.compute_throughput_ratios(init=False)

        # fix bsz for ncf
        cname = next(iter(self.num_nodes.keys()))
        for job in self.jobs:
            if job.applications[cname].name == "ncf":
                job.target_batch_size = 32768

        # map to pmp jobs
        random.seed(0)
        if self.allow_pmp:
            for job in self.jobs:
                if 'bert' in job_name_map and job.app_name == 'bert':
                    p = random.random()
                    if p < 1/3:
                        job.real_job_name = 'gpt-1.3b'
                        if isinstance(self.policy, GavelPolicy): job.target_num_replicas = 8
                    elif p < 2/3:
                        job.real_job_name = 'gpt-6.7b'
                        if isinstance(self.policy, GavelPolicy): job.target_num_replicas = 16
                    else:
                        job.real_job_name = 'gpt-15b'
                        if isinstance(self.policy, GavelPolicy): job.target_num_replicas = 32
                    # input(f'p: {p} job: {job.real_job_name}')

    def step(self, seconds=60):
        start_t = time.time()
        self.step_jobs(seconds)
        end_t = time.time()
        print(f"Job Step time: {(end_t - start_t)*1000}ms")
        self.optimize_policy()
        end_t2 = time.time()
        print(f"Scheduling compute time: {(end_t2 - end_t)*1000}ms")

        # update cluster goodput ratios
        # if isinstance(self.policy, WeightedMIPPolicy):
        #    self.update_througput_ratios(self.policy.get_current_gput_ratios())
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

            #  check batch size
            if isinstance(self.policy, WeightedMIPFIXPolicy):
                error = 1 * (job.accum_steps + 1) * int(sum(job.placement))
                actual_bsz = job.atomic_bsz * (job.accum_steps + 1) * int(sum(job.placement))
                assert actual_bsz == 0 or abs(actual_bsz - job.target_batch_size) <= error, f"actual: {actual_bsz}, target: {job.target_batch_size}"
            
        self.current_time += seconds
    
    def optimize_policy(self):
        job_infos = self.get_job_infos()
        if job_infos:
            # Optimize allocations.
            node_infos = self.get_node_infos()
            self.allocations = {k: v for k, v in self.allocations.items() if k in job_infos}
            
            # run scheduling policy
            p_start_t = time.time()
            allocations = self.policy.optimize(job_infos, node_infos, self.allocations)
            p_end_t = time.time()
            print(f"policy.optimize(): {round((p_end_t-p_start_t)*1000)}ms")
            
            factor = 0   
            for job in self.jobs:
                if job.submission_time <= self.current_time and job.completion_time is None:
                    # record load factor
                    if isinstance(self.policy, GavelPolicy):
                        factor += job.target_num_replicas

                    job.contention.append(len(job_infos))
                    # print(f"job: {job.name}, contention: {job.contention}")
                    if isinstance(self.policy, WeightedMIPPolicy) or isinstance(self.policy, GavelPolicy) or isinstance(self.policy, WeightedMIPFIXPolicy):
                        if isinstance(self.policy, WeightedMIPPolicy):
                            assert job.name in allocations, f'{job.name} not in the allcation returned by the policy'

                        job_allocation = allocations.get(job.name, None)
                        if self.allow_pmp and job.app_name in job_name_map: # NOTE: handle pmp
                            config = None if job_allocation is None else {k: len(v) for k,v in job_allocation.items()}
                            # add config to job history
                            if config is not None:
                                key = tuple([0 if cname not in config else config[cname] for cname in self.clusters])
                                if key not in job.pmp_config_history:
                                    job.pmp_config_history[key] = 0
                                job.pmp_config_history[key] += 60 if config == job.pmp_config else 15 # NOTE: change this when using differnt intervals
                            job.allocate(config, job_allocation)
                        else:
                            if job_allocation is None:
                                cluster = None
                                alloc = ()
                            else:
                                alloc = [(k,v) for k,v in job_allocation.items()]
                                cluster, alloc = alloc[0][0], alloc[0][1]
                            if isinstance(self.policy, WeightedMIPFIXPolicy):
                                assert len(alloc) == 0 or len(alloc) == job.target_num_replicas, f'{job.name} {job.target_num_replicas} {alloc}'
                            # change in resources
                            if allocations.get(job.name) != self.allocations.get(job.name, None):
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

                    elif isinstance(self.policy, PmpPolicy):
                        alloc = allocations.get(job.name, None)
                        job.allocate(alloc)
                    else:
                        assert False, "other policies not implemented"
                    # record rescheduling overhead
                    job.overhead = job.rescale_time
            # make a copy of allocations
            self.allocations = allocations

            if isinstance(self.policy, GavelPolicy):
                self.load_factors.append(factor)


    def compute_throughput_ratios(self, init = False):
        # throughput conversion ratios
        self.cluster_throughput_ratios = dict()
        for app_name in self.cluster_applications.keys():
            app_xput_ratios = dict()
            for cluster in self.clusters:
                ratios = dict()
                for dest_cluster in self.clusters:
                    # ignore any clusters that dont have speedup_fn ready
                    if dest_cluster == cluster:
                        continue
                    elif not init:
                        dest_df = self.cluster_applications[app_name][dest_cluster].placements
                        src_df = self.cluster_applications[app_name][cluster].placements
                        src_df = src_df[src_df.num_replicas == 1]
                        dest_df = dest_df[dest_df.num_replicas == 1]
                        mean_src_atomic_xput = (src_df.local_bsz.max() / src_df.step_time.max())
                        mean_dest_atomic_xput = (dest_df.local_bsz.max() / dest_df.step_time.max())
                        ratio = mean_src_atomic_xput / mean_dest_atomic_xput
                        ratios[dest_cluster] = ratio
                    else:
                        ratios[dest_cluster] = 1.0
                app_xput_ratios[cluster] = ratios
            self.cluster_throughput_ratios[app_name] = app_xput_ratios
        # print(f"Throughput ratios: {self.cluster_throughput_ratios}")
    
    # avg with the current value
    # self.cluster_throughput_ratios = {model : {dest_cluster : {src_cluster : ratio}}}
    # new_xput_ratios: {dest_cluster : {src_cluster : ratio}}
    def update_througput_ratios(self, new_xput_ratios):
        print(f"Throughput ratios: {self.cluster_throughput_ratios}")
        return
        for dst_cluster in new_xput_ratios.keys():
            for src_cluster, val in new_xput_ratios[dst_cluster].items():
                for model in self.cluster_throughput_ratios.keys():
                    v2 = self.cluster_throughput_ratios[model][dst_cluster][src_cluster]
                    self.cluster_throughput_ratios[model][dst_cluster][src_cluster] = (val + v2) / 2

    def step_log(self):        
        step_log = {
            "timestamp": self.current_time,
            "num_nodes": self.num_nodes,
            "submitted_jobs": [
                {
                    "name": job.name,
                    "real_job_name": job.real_job_name if job.real_job_name is None else job.real_job_name+'-'+job.name.split('-')[-1],
                    "epoch": job.epoch,
                    "progress": job.progress,
                    "num_restarts": job.num_restarts,
                    "cluster": job.current_cluster,
                    "allocation": self.allocations.get(job.name, None),
                    # "allocation": None if job.name not in self.allocations else {cname: len(alloc) for cname, alloc in self.allocations[job.name].items()},
                    "placement": ({job.current_cluster: job.placement} if job.current_cluster is not None else {}) if job.app_name not in job_name_map or not self.allow_pmp else job.placement,
                    "batch_size": job.atomic_bsz * (job.accum_steps + 1) * int(sum(job.placement)) if job.app_name not in job_name_map or not self.allow_pmp else job.target_batch_size,
                    # "batch_size": job.batch_size,
                    "accum_steps": job.accum_steps,
                    "submission_time": job.submission_time,
                    "start_time": job.start_time,
                    "completion_time": job.completion_time,
                    "n_avg" : np.mean(np.asarray(job.contention)),
                    "overhead": job.overhead
                }
                for job in self.jobs if job.submission_time <= self.current_time
            ],
        }
        self.logs.append(step_log)

    def get_pmp_config_history(self):
        res = {}
        for job in self.jobs:
            if job.app_name in job_name_map:
                res[job.real_job_name+'-'+job.name.split('-')[-1]] = job.pmp_config_history
        return res

        
    def get_progress_track(self):
        res = {}
        for job in self.jobs:
            res[job.name] = job.progress_track
        return res


    def get_job_infos(self):
        job_infos = {}
        for job in self.jobs:
            if self.current_time >= job.submission_time and job.completion_time is None:
                if isinstance(self.policy, WeightedMIPPolicy):
                    job_infos[job.name] = self.get_weighted_mip_multi_job_info(job)
                elif isinstance(self.policy, GavelPolicy):
                    job_infos[job.name] = self.get_gavel_job_info(job)
                elif isinstance(self.policy, WeightedMIPFIXPolicy):
                    job_infos[job.name] = self.get_weighted_mip_fix_multi_job_info(job)
                elif isinstance(self.policy, PmpPolicy):
                    job_infos[job.name] = self.get_pmp_job_info(job)
                else:
                    job_infos[job.name] = self.get_pollux_job_info(job)
        return job_infos

    def get_pollux_job_info(self, job):
        job_info = JobInfo(
            resources={"nvidia.com/gpu": 1},
            speedup_fn=job.get_speedup_fn(),
            creation_timestamp=job.submission_time,
            attained_service=job.attained_service,
            min_replicas=0,
            max_replicas=min(max(2 * job.max_profiled_replicas, 1), 64,  # simulator can't handle more.
                             job.application.max_batch_size // job.application.min_local_bsz),
            preemptible=True,
        )
        if job.application.name == "ncf":
            job_info.max_replicas = 1
        job_info.num_restarts = job.num_restarts or 0
        job_info.age = self.current_time - job.submission_time
        return job_info

    def get_weighted_mip_multi_job_info(self, job):
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
            max_replicas= max_replicas,
            preemptible=True,
        )
        
        for cname, capp in job.applications.items():
            if capp.name == "ncf":
                job_info.max_replicas[cname] = 1
        job_info.num_restarts = job.num_restarts or 0
        job_info.num_migrations = job.num_migrations or 0
        job_info.age = self.current_time - job.submission_time
        job_info.real_job_name = job.real_job_name
        app_name = job.name.split('-')[0]
        job_info.app_name = app_name
        job_info.efficiency = job.compute_efficiency()

        # throughput conversion ratios
        job_info.cluster_throughput_ratios = self.cluster_throughput_ratios[app_name]

        return job_info

    def get_weighted_mip_fix_multi_job_info(self, job):
        speedup_fns = {cname : job.get_speedup_fn(cname) for cname in self.clusters}
        target_replicas = {cname : job.target_num_replicas for cname in self.clusters}
        job_info = JobInfo(
            resources={"nvidia.com/gpu": 1},
            speedup_fn=speedup_fns,
            creation_timestamp=job.submission_time,
            attained_service=job.attained_service,
            min_replicas= target_replicas,
            max_replicas= target_replicas,
            preemptible=True,
        )
        
        for cname, capp in job.applications.items():
            if capp.name == "ncf":
                job_info.max_replicas[cname] = 1
        job_info.num_restarts = job.num_restarts or 0
        job_info.num_migrations = job.num_migrations or 0
        job_info.age = self.current_time - job.submission_time
        # job_info.target_batch_size = job.target_batch_size

        app_name = job.name.split('-')[0]

        # throughput conversion ratios
        job_info.cluster_throughput_ratios = self.cluster_throughput_ratios[app_name]

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
        job_info.real_job_name = job.real_job_name
        job_info.epoch = job.epoch
        job_info.target_batch_size = job.target_batch_size
        job_info.scale_factor = job.target_num_replicas
        job_info.age = self.current_time - job.submission_time
        return job_info

    def get_pmp_job_info(self, job):
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
        efficiency = job.compute_efficiency(),

        job_info.epoch = job.epoch
        job_info.target_batch_size = job.target_batch_size
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
                val["name"] if val["real_job_name"] is None else val["real_job_name"]: val["completion_time"] - val["submission_time"]
                for val in self.logs[-1]["submitted_jobs"]
                if val["completion_time"] is not None 
               }
        else:
            return {}

    def get_num_restarts(self):
        if len(self.logs) > 0:
            return {
                val["name"] if val["real_job_name"] is None else val["real_job_name"]: val["num_restarts"]
                for val in self.logs[-1]["submitted_jobs"]
                if val["completion_time"] is not None 
               }
        else:
            return {}