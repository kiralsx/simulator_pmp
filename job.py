from email.mime import application
import math
from nis import cat
from re import A, L
import time
from tkinter import E
import numpy as np
import os
from pathlib import Path
import json

from applications import APPLICATIONS
from goodput import GoodputFunction, fit_perf_params, PerfParams, GradParams
from speedup import SpeedupFunction, UncachedSpeedupFunction
from utils import JobInfo, NodeInfo, job_name_map, cluster_name_map, order_cluster

class Job(object):
    def __init__(self, name, applications, submission_time,
                 target_num_replicas=None, target_batch_size=None, 
                 cache_speedups=False, h_unaware=False, app_name=None, allow_pmp=False):
        self.name = name
        self.h_unaware = h_unaware
        # applications is a dict w/ key=cluster_name, val=cluster_specific_App
        self.applications = applications
        self.submission_time = submission_time
        self.target_num_replicas = target_num_replicas
        self.target_batch_size = target_batch_size
        self.start_time = None
        self.completion_time = None
        self.current_time = 0
        self.rescale_time = 0
        self.atomic_bsz = 0
        self.accum_steps = 0
        # profile is also a dict w/ key=cluster_name, val=cluster_specific_profile
        # if h_unaware is True, profiles is just all profiles merged into one dict
        self.profiles = dict()
        # perf_params is also a dict w/ key=cluster_name, val=cluster_specific_perf_params
        if self.h_unaware:
            self.perf_params = None
        else:
            self.perf_params = dict()
            for cluster in self.applications.keys():
                self.profiles[cluster] = dict()
                self.perf_params[cluster] = None
        # grad_params is common to all clusters (related to progress of job)
        self.grad_params = None
        self.best_metric = None
        self.progress = 0.0
        self.epoch = 0
        self.attained_service = 0
        self.contention = []
        self.num_restarts = None
        self.num_migrations = None
        self.migrate_penalty = 45
        self.rescale_penalty = 30

        # cluster that this job is allocated to
        self.current_cluster = None
        if cache_speedups:
            self.speedup_fn_class = SpeedupFunction
        else:
            self.speedup_fn_class = UncachedSpeedupFunction
        print(f"SPEEDUP_FN_CLASS: {self.speedup_fn_class.__name__}")

        iter_time_cache_path = os.path.join(os.path.dirname(__file__), f'iteration_time.txt')
        if Path(iter_time_cache_path).is_file():
            self.iter_time_dict = json.load(open(iter_time_cache_path))
        else:   
            raise FileNotFoundError('iteration time matrix is not found')
        
        self.cluster_name = order_cluster(list(cluster_name_map.keys()))
        self.app_name = app_name 
        self.allow_pmp = allow_pmp
        self.pmp_config = None
        self.pmp_config_history = dict()

        if not self.allow_pmp or self.app_name not in job_name_map:
            self.real_job_name = None
            self.placement = ()
        else:
            self.real_job_name = job_name_map[self.app_name]
            self.placement = {}

        self.overhead = 0

        # self.progress_track = []
    def seed_profiles(self, max_num_nodes, max_num_replicas):
        print(f"Seeding profiles for job: {self.name}")
        for cluster, cluster_app in self.applications.items():
            self.profiles[cluster] = dict()
            profile = self.profiles[cluster]

            # add placements data
            if max_num_nodes > 0:
                placements_selector = (cluster_app.placements.num_nodes <= max_num_nodes) & (cluster_app.placements.num_replicas <= max_num_replicas)
                df = cluster_app.placements[placements_selector]
            else:
                df = cluster_app.placements

            num_nodes, num_replicas, local_bsz, step_time, sync_time = df.num_nodes.to_numpy(), df.num_replicas.to_numpy(), df.local_bsz.to_numpy(), df.step_time.to_numpy(), df.sync_time.to_numpy()
            for i in range(len(num_nodes)):
                self.profiles[cluster][num_nodes[i], num_replicas[i], local_bsz[i]] = step_time[i], sync_time[i]
            # add scalability data
            if max_num_nodes > 0:
                scalability_selector = (cluster_app.scalability.num_nodes <= max_num_nodes) & (cluster_app.scalability.num_replicas <= max_num_replicas)
                df = cluster_app.scalability[scalability_selector]
            else:
                df = cluster_app.scalability

            num_nodes, num_replicas, local_bsz, step_time, sync_time = df.num_nodes.to_numpy(), df.num_replicas.to_numpy(), df.local_bsz.to_numpy(), df.step_time.to_numpy(), df.sync_time.to_numpy()
            for i in range(len(num_nodes)):
                profile_key = (num_nodes[i], num_replicas[i], local_bsz[i])
                if profile_key not in profile:
                    profile[profile_key] = step_time[i], sync_time[i]

            # update perf params for cluster
            num_nodes = np.array([key[0] for key in profile])
            num_replicas = np.array([key[1] for key in profile])
            local_bsz = np.array([key[2] for key in profile])
            step_time = np.array([val[0] for val in profile.values()])
            sync_time = np.array([val[1] for val in profile.values()])
            compute_time = step_time - sync_time
            
            cluster_perf_params = fit_perf_params(num_nodes, num_replicas, local_bsz, compute_time, step_time)
            self.perf_params[cluster] = cluster_perf_params

    def max_profiled_replicas(self, cluster_name=None):
        if not cluster_name and self.h_unaware:
            return max((k[1] for k in self.profiles), default=0)
        if cluster_name in self.profiles:
            return max((k[1] for k in self.profiles[cluster_name]), default=0)
        else:
            return 0

    def get_goodput_fn(self, cluster_name=None):
        app = self.applications[cluster_name if cluster_name else "aws"]
        if self.h_unaware:
            perf_params, grad_params = self.perf_params, self.grad_params
        else:
            perf_params, grad_params = self.perf_params[cluster_name], self.grad_params
            
        # no throughput model yet
        if grad_params is None or perf_params is None:
            return None
        else:
            return GoodputFunction(perf_params, grad_params, app.init_batch_size)

    def compute_efficiency(self):
        if not self.allow_pmp or not self.app_name in job_name_map:
            return None
        application = self.applications[list(self.applications.keys())[0]]
        scale = self.target_batch_size / application.init_batch_size # TODO: check if need to modify the init_batch_size
        grad_sqr, grad_var = application.get_grad_stats(self.target_batch_size, self.epoch)
        gain = (grad_var + grad_sqr) / (grad_var / scale + grad_sqr)
        return gain

    def get_speedup_fn(self, cluster_name=None):
        if self.h_unaware:
            if self.grad_params is None:
                return lambda n, r : r
        else:
            perf_params = self.perf_params[cluster_name]
            if self.grad_params is None or perf_params is None:
                return None
        app = self.applications[cluster_name if cluster_name else "aws"]
        return self.speedup_fn_class(self.get_goodput_fn(cluster_name), app.max_batch_size if self.target_batch_size is None else self.target_batch_size,
                               (app.min_local_bsz, app.max_local_bsz),
                               accumulation=True,
                               fix_batch_size = self.target_batch_size is not None)

    # fixes batch size to obey memory/profiling constraints
    # needed for gavel
    def fix_minibatch_size(self):
        if self.allow_pmp and self.app_name in job_name_map:
            return
        if self.target_num_replicas is not None and self.target_batch_size is not None:
            max_atomic_bsz = math.ceil(self.target_batch_size / self.target_num_replicas - 1e-8)
            for cluster, cluster_app in self.applications.items():
                if self.target_num_replicas in cluster_app.placements.num_replicas.values:
                    df = cluster_app.placements[cluster_app.placements.num_replicas == self.target_num_replicas]
                    new_bsz = int(min(max_atomic_bsz, df.local_bsz.max()))
                    if new_bsz < max_atomic_bsz:
                        print(f"{self.name}: correcting atomic_bsz: {max_atomic_bsz} -> {new_bsz}")
                        max_atomic_bsz = new_bsz
            target_batch_size = self.target_num_replicas * max_atomic_bsz
            self.target_batch_size = min(self.target_batch_size, target_batch_size)

    def update_local_bsz(self, placement):
        if self.allow_pmp and self.app_name in job_name_map:
            raise Exception(f'{self.name}: should not update local batch size')
        if self.current_cluster is None:
            assert False, "updating local bsz before assigning cluster"
        app = self.applications[self.current_cluster]
        placement = tuple(filter(None, placement))
        num_nodes, num_replicas = len(placement), sum(placement)
        batch_size = self.target_batch_size
        perf_params = self.perf_params if self.h_unaware else self.perf_params[self.current_cluster]
        grad_params = self.grad_params
        max_local_bsz = app.get_max_local_bsz(placement)
        if batch_size is None and (grad_params is None or perf_params is None):
            batch_size = max(app.init_batch_size, app.min_local_bsz * num_replicas)
        if batch_size is None:
            goodput_fn = self.get_goodput_fn(self.current_cluster)
            _, self.atomic_bsz, self.accum_steps = goodput_fn.optimize(
                num_nodes, num_replicas, app.max_batch_size,
                (app.min_local_bsz, max_local_bsz), accumulation=True)
        else:
            local_bsz = math.ceil(batch_size / num_replicas - 1e-8)
            try:
                self.accum_steps = math.ceil(local_bsz / max_local_bsz - 1e-8) - 1
            except Exception:
                print(f'{self.name} {self.real_job_name} {self.target_batch_size} {local_bsz} {max_local_bsz}')
                exit()
            if num_replicas == 1 and batch_size > app.init_batch_size:
                self.accum_steps = max(1, self.accum_steps)
            self.atomic_bsz = math.ceil(local_bsz / (self.accum_steps + 1) - 1e-8)
        
        # correct self.atomic_bsz to take into account memory constraints
        if num_replicas in app.placements.num_replicas.values:
            df = app.placements[app.placements.num_replicas == num_replicas]
            new_bsz = int(min(self.atomic_bsz, df.local_bsz.max()))
            if new_bsz < self.atomic_bsz:
                print(f"{self.name}: correcting atomic_bsz: {self.atomic_bsz} -> {new_bsz}")
                self.atomic_bsz = new_bsz
        
        count = num_replicas * (self.accum_steps + 1)
        self.atomic_bsz = min(self.atomic_bsz, int(app.max_batch_size / count))

        # additional check to see if it uses larger bsize on dgx
        if False:
            app_name = app.name
            dgx_app = APPLICATIONS['dgx'][app_name]
            max_dgx_placements_df = dgx_app.placements[dgx_app.placements.num_replicas == 16]
            max_atomic_bsz = max_dgx_placements_df.local_bsz.max()
            if self.atomic_bsz > max_atomic_bsz:
                print(f"MAX ATOMIC BSZ EXCEEDED: {self.name}, max={max_atomic_bsz}, cur={self.atomic_bsz}")


    def update_params(self, num_nodes, num_replicas, local_bsz,
                      step_time, sync_time, grad_sqr, grad_var):
        if self.allow_pmp and self.app_name in job_name_map:
            raise Exception(f'{self.name}: should not update_params')
        assert self.current_cluster is not None, "current_cluster is None??"
        self.grad_params = (grad_sqr, grad_var)
        if self.h_unaware:
            profile = self.profiles
        else:
            profile = self.profiles[self.current_cluster]
        
        if (num_nodes, num_replicas, local_bsz) in profile:
            return
        
        if self.h_unaware:
            self.profiles[num_nodes, num_replicas, local_bsz] = step_time, sync_time
        else:
            self.profiles[self.current_cluster][num_nodes, num_replicas, local_bsz] = step_time, sync_time

        num_nodes = np.array([key[0] for key in profile])
        num_replicas = np.array([key[1] for key in profile])
        local_bsz = np.array([key[2] for key in profile])
        step_time = np.array([val[0] for val in profile.values()])
        sync_time = np.array([val[1] for val in profile.values()])
        compute_time = step_time - sync_time
        
        perf_params = fit_perf_params(num_nodes, num_replicas, local_bsz, compute_time, step_time)
        if self.h_unaware:
            self.perf_params = perf_params
        else:
            self.perf_params[self.current_cluster] = perf_params

    def step_pmp(self, seconds, interference=0.0):
        if self.pmp_config is None:
            self.current_time += seconds
            return
    
        delay = min(self.rescale_time, seconds)
        self.current_time += delay
        # self.attained_service += delay * sum(self.placement) # TODO: check what attained_service is used for
        self.rescale_time -= delay
        seconds -= delay
        while seconds > 0 and self.completion_time is None:
            # throughput
            alloc_key = ""
            for cname in self.cluster_name:
                if cname not in self.pmp_config:
                    alloc_key += f'0_{cluster_name_map[cname]}_'
                else:
                    alloc_key += f'{self.pmp_config[cname]}_{cluster_name_map[cname]}_'
            alloc_key = alloc_key[:-1]
            total_time = self.iter_time_dict[self.real_job_name][alloc_key]
            # total_time = 0.01

            # stats_effciency
            application = self.applications[list(self.applications.keys())[0]]
            gain = self.compute_efficiency()

            # goodput
            goodput = gain / total_time * (1.0 - interference)

            # print(f"#### gain: {gain} total_time: {total_time} goodput: {goodput}")

            # Update current epoch and progress.
            next_progress = application.get_progress(self.epoch + 1)
            if self.progress + goodput * seconds < next_progress:
                # Used up the entire time interval without finishing an epoch.
                self.progress += goodput * seconds
                self.current_time += seconds
                # self.attained_service += seconds * sum(self.placement)
                seconds = 0
            else:
                # Crossed an epoch boundary before finishing the time interval.
                self.epoch += 1
                delta = round(float((next_progress - self.progress) / goodput))
                assert delta <= seconds
                completion_epoch = application.get_completion_epoch(self.target_batch_size)
                if self.epoch > completion_epoch:
                    self.completion_time = self.current_time + delta
                self.progress = next_progress
                self.best_metric = application.get_best_metric(self.target_batch_size, self.epoch)
                self.current_time += delta
                # self.attained_service += delta * sum(self.placement)
                seconds -= delta
                # Re-scale batch size between epochs.
        self.current_time += seconds


    def step(self, seconds, interference=0.0):
        if self.allow_pmp and self.app_name in job_name_map:
            return self.step_pmp(seconds, interference)
        if not self.placement:
            # No resources are allocated to this job.
            self.current_time += seconds
            return
        delay = min(self.rescale_time, seconds)
        self.current_time += delay
        self.attained_service += delay * sum(self.placement)
        self.rescale_time -= delay
        seconds -= delay
        while seconds > 0 and self.completion_time is None:
            assert self.current_cluster is not None, "stepping on job without current_cluster set"
            application = self.applications[self.current_cluster]
            assert self.epoch < application.max_epochs
            # print(f"job: {self.name}, placement: {self.placement}")
            # Calculate current job configurations.
            placement = tuple(filter(None, self.placement))
            num_nodes, num_replicas = len(placement), sum(placement)
            local_bsz = self.atomic_bsz
            batch_size = num_replicas * self.atomic_bsz * (self.accum_steps + 1)
            scale = batch_size / application.init_batch_size
            # Calculate true (simulated) throughput.
            step_time, sync_time = application.get_throughput(placement, self.atomic_bsz)
            accum_time = step_time - sync_time
            # Calculate true (simulated) efficiency.
            grad_sqr, grad_var = application.get_grad_stats(batch_size, self.epoch)
            gain = (grad_var + grad_sqr) / (grad_var / scale + grad_sqr)
            # Update the estimated throughput/efficiency parameters.
            self.update_params(num_nodes, num_replicas, self.atomic_bsz,
                               step_time, sync_time, grad_sqr, grad_var)
            # Calculate true (simulated) goodput.
            total_time = step_time + accum_time * self.accum_steps
            # total_time = 0.01
            # print(f'#### total_time: {total_time} accum_steps: {self.accum_steps}')
            goodput = gain / total_time * (1.0 - interference)

            # goodput multiplier
            # goodput = self.multiplier * goodput

            # Update current epoch and progress.
            next_progress = application.get_progress(self.epoch + 1)
            if self.progress + goodput * seconds < next_progress:
                # Used up the entire time interval without finishing an epoch.
                self.progress += goodput * seconds
                self.current_time += seconds
                self.attained_service += seconds * sum(self.placement)
                seconds = 0
            else:
                # Crossed an epoch boundary before finishing the time interval.
                self.epoch += 1
                delta = round(float((next_progress - self.progress) / goodput))
                assert delta <= seconds
                completion_epoch = application.get_completion_epoch(batch_size)
                if self.epoch > completion_epoch:
                    self.completion_time = self.current_time + delta
                self.progress = next_progress
                self.best_metric = application.get_best_metric(batch_size, self.epoch)
                self.current_time += delta
                self.attained_service += delta * sum(self.placement)
                seconds -= delta
                # Re-scale batch size between epochs.
            self.update_local_bsz(self.placement)
        self.current_time += seconds  # Add any remaining time.


    def reallocate(self, placement):
        if self.allow_pmp and self.app_name in job_name_map:
            raise Exception(f'{self.name}: should not reallocate')
        
        if placement:
            if self.placement != tuple(placement):
                print(f"RESCALE: job: {self.name}, cluster: {self.current_cluster}, placement: {self.placement} -> {placement}")
                self.placement = tuple(placement)
                self.update_local_bsz(self.placement)
                # Start startup/re-scale countdown.
                self.rescale_time = self.rescale_penalty
                if self.num_restarts is None:
                    self.num_restarts = 0
                else:
                    self.num_restarts += 1
        else:  # De-allocate all resources.
            print(f"SUSPEND: job: {self.name}")
            self.placement = ()
            self.atomic_bsz = 0
            
    
    def migrate(self, new_cluster, new_placement):
        if self.allow_pmp and self.app_name in job_name_map:
            raise Exception(f'{self.name}: should not migrate')
        # set current cluster
        prev_cluster = self.current_cluster
        print(f"MIGRATE:: {self.name}, cluster: {prev_cluster} -> {new_cluster}")
        self.current_cluster = new_cluster
        if new_placement:
            self.placement = tuple(new_placement)
            self.update_local_bsz(self.placement)
            # Start startup/re-scale countdown.
            self.rescale_time = self.migrate_penalty  
            if self.num_restarts is None:
                self.num_restarts = 0
            else:
                self.num_restarts += 1
        else:
            print(f"SUSPEND: job: {self.name}")
            # De-allocate all resources.
            self.placement = ()
            self.atomic_bsz = 0

    
    def allocate(self, config, alloc):
        if not self.allow_pmp:
            raise Exception(f'{self.name}: when pmp is not allowed, should not allocate')
        if config != self.pmp_config:
            if config is None and self.pmp_config is not None:
                print(f'SUSPEND job: {self.name}')
            else:
                if config is not None and self.pmp_config is None:
                    print(f'RESTART job: {self.name}')
                else:
                    print(f'REALLOCATE job: {self.name}, prev: {self.pmp_config}, new: {config}')
                if self.num_restarts is None:
                    self.num_restarts = 0
                else:
                    self.num_restarts += 1
                self.rescale_time = self.migrate_penalty

            # update alloc and placement
            self.pmp_config = config
            self.placement = {}
            if config is not None:
                for cluster_name, cluster_alloc in alloc.items():
                    placement = []
                    for i in range(len(cluster_alloc)):
                        if i == 0 or cluster_alloc[i] != cluster_alloc[i - 1]:
                            placement.append(1)
                        else:
                            placement[-1] += 1
                    self.placement[cluster_name] = placement
        else:
            if config is None:    
                print(f'IDLE job: {self.name}')
            else:
                print(f'NOCHANGE job: {self.name}')

