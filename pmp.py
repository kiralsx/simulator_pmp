import itertools
from pathlib import Path
import json
import numpy as np
import cvxpy as cp
import collections
import copy
import math
import os
# from mip_cvxpy import PYTHON_MIP

cluster_name_map = {'rtx' : 'rtx', 'dgx-ext' : 'a100'}
job_name_map = {'imagenet': 'wresnet-2b'}


class PmpPolicy(object):
    def __init__(self, interval, homoegeneous=False):
        # self.cluster_list = slice_cluster()
        self.homogeneous = homoegeneous

        # prepare throughput matrix for each job
        self.supported_jobs = ['wresnet-13b', 'gpt-6.7b', 'gpt-1.3b', 'gpt-15b']
        self.num_micro_batches_dict = {'wresnet-13b': 38, 'wresnet-6.5b': 38, 'wresnet-2b': 24, 'gpt-6.7b': 128, 'gpt-1.3b': 128, 'gpt-15b':128}
        iter_time_cache_path = os.path.join(os.path.dirname(__file__), f'iteration_time.txt')
        if Path(iter_time_cache_path).is_file():
            self.iter_time_dict = json.load(open(iter_time_cache_path))
        else:   
            raise FileNotFoundError('iteration time matrix is not found')

        self.worker_assignments = collections.OrderedDict()
        self.allocs = {}


    def slice_cluster(self, cluster, homogeneous=True):
        single = {cname: [] for cname in cluster}
        for cname, num_gpu in cluster.items():
            i = 0
            while i <= num_gpu:
                single[cname].append(i)
                if i == 0:
                    i = i + 1
                elif i <= 8:
                    i = i * 2
                else:
                    i = i + 8
        res = list(itertools.product(*list(single.values())))[1:]
        if homogeneous:
            res = [e for e in res if e[0] == 0 or e[1] == 0]
        # res = [{list(cluster.keys())[0]: e[0], list(cluster.keys())[1]: e[1]} for e in res]
        return res


    def populate_valid_configs(self, cluster_num_nodes, cluster_num_gpus):
        self.num_gpus_per_server = {cluster_name_map[cname]: cluster_num_gpus[cname] for cname in cluster_num_gpus}
        assert len(cluster_num_gpus) == 2, f'can only support two gpu types'
        self.cluster = {cluster_name_map[cname]: (cluster_num_nodes[cname] * cluster_num_gpus[cname]) for cname in cluster_num_nodes}
        self.cluster_name = list(self.cluster.keys())
        self.valid_allocs = self.slice_cluster(cluster=self.cluster, homogeneous=self.homogeneous)
        print(f'valid_alloc: {self.valid_allocs}')

        self.worker_id_to_cluster_mapping = {}
        i = 0
        for cname in self.cluster_name:
            for _ in range(self.cluster[cname]):
                self.worker_id_to_cluster_mapping[i] = cname
                i += 1

        self.cluster_to_worker_id_mapping = {}
        j = 0
        n = 0
        for c in cluster_num_nodes:
            cname = cluster_name_map[c]
            self.cluster_to_worker_id_mapping[cname] = []
            num_gpu_per_server = cluster_num_gpus[c]
            num_server = cluster_num_nodes[c]
            for i in range(num_server):
                self.cluster_to_worker_id_mapping[cname].append(list(range(n + num_gpu_per_server*i, n + num_gpu_per_server*(i+1))))
            j += 1
            n += self.cluster[cname]
        
        print("### worker_id_to_cluster_mapping")
        print(self.worker_id_to_cluster_mapping)
        print("### cluster_to_worker_id_mapping")
        print(self.cluster_to_worker_id_mapping)


    def convert_worker_ids(self, cname, worker_ids):
        res = []
        for worker in worker_ids:
            for cid in range(len(self.cluster_name)):
                if cname == self.cluster_name[cid]:
                    idx = worker - sum([self.cluster[cname] for cname in self.cluster_name[:cid]])
                    res.append(math.floor(idx / self.num_gpus_per_server[cname]))
                    # res.append(worker - sum([self._cluster_spec[cname] for cname in self._cluster_name[:cid]]))
        return res


    def convert_app_name(self, info):
        return job_name_map[info.applications[list(cluster_name_map.keys())[0]].name]


    def solve(self, job_info, iter_time_dict, valid_allocs):
        job_list = list(job_info.keys())
        app_list = [self.convert_app_name(info) for info in job_info.values()]
        # print(f"job_list: {job_list}")
        print(f"app_list: {app_list}")

        def get_max_min_share(throughput_dict, gpu_id=0):
            alloc = -1
            for app in app_list:
                throughput_matrix = throughput_dict[app]
                can_run = False
                for num in [c[gpu_id] for c in valid_allocs if c[(gpu_id+1)%2] == 0]:
                    if gpu_id == 0:
                        if throughput_matrix[f'{num}_{self.cluster_name[0]}_0_{self.cluster_name[1]}'] != np.inf:
                            can_run = True
                            break
                    else:
                        if throughput_matrix[f'0_{self.cluster_name[0]}_{num}_{self.cluster_name[1]}'] != np.inf:
                            can_run = True
                            break
                alloc = max(alloc, num)
                assert can_run == True
            assert alloc > 0
            return alloc

        norm_gpu_id = 0
        max_min_share = get_max_min_share(iter_time_dict, norm_gpu_id)     

        share_key = f'{max_min_share}_{self.cluster_name[0]}_0_{self.cluster_name[1]}' if norm_gpu_id == 0 else f'0_{self.cluster_name[0]}_{max_min_share}_{self.cluster_name[1]}'
        # print(f'share throughput: {(1 / iter_time_dict[app_list[0]][share_key])}')
        # exit()
        x = cp.Variable((len(app_list), len(valid_allocs)), boolean=True)
        t = np.zeros((len(app_list), len(valid_allocs)), dtype=float)
        t_real = np.zeros((len(app_list), len(valid_allocs)), dtype=float)

        for i in range(len(app_list)):
            for j in range(len(valid_allocs)):
                c = valid_allocs[j]
                key = f'{c[0]}_{self.cluster_name[0]}_{c[1]}_{self.cluster_name[1]}'
                if iter_time_dict[app_list[i]][key] is None:
                    t[i][j] = 1e-9
                    t_real[i][j] = 0
                else:
                    t_real[i][j] = (1 / iter_time_dict[app_list[i]][key])
                    share_key = f'{max_min_share}_{self.cluster_name[0]}_0_{self.cluster_name[1]}' if norm_gpu_id == 0 else f'0_{self.cluster_name[0]}_{max_min_share}_{self.cluster_name[1]}'
                    t[i][j] = t_real[i][j] / (1 / iter_time_dict[app_list[i]][share_key])
                
                # force to choose the smallest allocation with same t value
                if t[i,j] in t[i, :j]:
                    t[i,j] = 1e-9

        # # normalize t
        # for i in range(t.shape[0]):
        #     t[i] = t[i] / np.min([v for v in t[i] if v != 0])
        
        p = 0.5
        t = np.power(t, p)        

        sum_throughput = cp.sum(cp.multiply(x, t))
        obj = cp.Maximize(sum_throughput)

        constraints = [cp.sum(x, axis=1, keepdims=True) <= 1]

        gpu_0 = []
        gpu_1 = []
        for i in range(len(app_list)):
            gpu_0.append([c[0] for c in valid_allocs])
            gpu_1.append([c[1] for c in valid_allocs])
        gpu_0 = np.array(gpu_0)
        gpu_1 = np.array(gpu_1)
        gpu_used_0 = cp.sum(cp.multiply(x, gpu_0))
        gpu_used_1 = cp.sum(cp.multiply(x, gpu_1))

        constraints.append(gpu_used_0 <= self.cluster[self.cluster_name[0]])
        constraints.append(gpu_used_1 <= self.cluster[self.cluster_name[1]])

        prob = cp.Problem(obj, constraints)
        # print(prob)
        # print('')

        # result = prob.solve(solver=PYTHON_MIP())
        result = prob.solve(solver=cp.GLPK_MI)

        if prob.status != "optimal":
            print('WARNING: Allocation returned by policy not optimal!')
        elif x.value is None:
            print('WARNING: No allocation possible with provided SLOs!')
        else:
            value = (x.value)
            allocs = {}
            # print(value)
            # print('--- allocation: ---')
            sum_t_real = []
            for i in range(len(app_list)):
                allocated = False
                for j in range(len(valid_allocs)):
                    if value[i,j] == 1:
                        # print(f'{job_list[i]}, {valid_allocs[j]}, norm: {t[i,j]}, throughpout: {t_real[i, j]}')
                        sum_t_real.append(t_real[i, j])
                        allocated = True
                        allocs[job_list[i]] = {self.cluster_name[0]: valid_allocs[j][0], self.cluster_name[1]: valid_allocs[j][1]}
                        
                        valid_allocs[j]
                        # break
                if not allocated:
                    sum_t_real.append(0)
                    # allocs[job_list[i]] = {}
                    # print(f'{job_list[i]}, None, througput: 0 no allocation')
            print(f'obj: {prob.value} sum throughput: {np.sum(sum_t_real)} std throughput: {np.std(sum_t_real)}')
            # print('------')
        return sum_t_real, prob.value, allocs


    def assign_workers_to_job(self, job_id, scale_factor, worker_state, worker_assignments):
        worker_ids = worker_state['worker_ids']
        assigned_worker_ids = worker_state['assigned_worker_ids']
        server_id_ptr = worker_state['server_id_ptr']

        if job_id in worker_assignments:
            worker_ids_for_job = list(worker_assignments[job_id])
        else:
            worker_ids_for_job = []
        while len(worker_ids_for_job) < scale_factor and server_id_ptr < len(worker_ids):
            if len(worker_ids[server_id_ptr]) == 0:
                server_id_ptr += 1
                continue
            worker_id_to_assign = worker_ids[server_id_ptr][0]
            if worker_id_to_assign not in assigned_worker_ids:
                worker_ids_for_job.append(worker_id_to_assign)
                assigned_worker_ids.add(worker_id_to_assign)
            worker_ids[server_id_ptr].pop(0)

        worker_assignments[job_id] = tuple(worker_ids_for_job)
        worker_state['server_id_ptr'] = server_id_ptr


    def schedule_jobs_on_workers(self, allocs):
        scheduled_jobs = {cname: [] for cname in self.cluster_name}
        for jname, alloc_ in allocs.items():
            for cname, num_gpu in alloc_.items():
                if num_gpu > 0:
                    scheduled_jobs[cname].append((jname, num_gpu))

        cluster_state = {}
        for cname in self.cluster_name:
            scheduled_jobs[cname].sort(key=lambda x: x[1], reverse=True)
            worker_ids = copy.deepcopy(self.cluster_to_worker_id_mapping[cname])
            cluster_state[cname] = {
                'worker_ids': worker_ids,
                'assigned_worker_ids': set(),
                'server_id_ptr': 0,
            }

        print(f'scheduled_jobs: {scheduled_jobs}')

        new_worker_assignments = {}

        # TODO: if a job assigned gpus in one cluster has already been changed,
        # don't need to maintain its allocation in other clusters
        for cname in self.cluster_name:
            print(f'cname: {cname}')
            new_worker_assignments[cname] = collections.OrderedDict()
            per_cluster_state = cluster_state[cname]
            assigned_worker_ids = per_cluster_state['assigned_worker_ids']

            scale_factors = set(x[1] for x in scheduled_jobs[cname])
            scale_factors = sorted(scale_factors, reverse=True)

            assert 0 not in scale_factors, f'0 should not be in scale_factors'

            for current_scale_factor in scale_factors:
                for (job_id, scale_factor) in scheduled_jobs[cname]:
                    if scale_factor != current_scale_factor:
                        continue
                    if job_id in self.allocs and cname in self.allocs[job_id] and self.allocs[job_id][cname] == scale_factor:
                        prev_worker_ids = self.worker_assignments[job_id][cname]
                        extend_placement = True
                        for prev_worker_id in prev_worker_ids:
                            if prev_worker_id in assigned_worker_ids:
                                extend_placement = False
                                break
                        if extend_placement:
                            new_worker_assignments[cname][job_id] = prev_worker_ids
                            for prev_worker_id in prev_worker_ids:
                                assigned_worker_ids.add(prev_worker_id)
                
                # Assign workers for remaining jobs.
                for job_id, scale_factor in scheduled_jobs[cname]:
                    if scale_factor != current_scale_factor:
                        continue
                    self.assign_workers_to_job(job_id, scale_factor,
                                                per_cluster_state,
                                                new_worker_assignments[cname])
        new_worker_assignments_combine = {}
        for cname in new_worker_assignments:
            for job_id in new_worker_assignments[cname]:
                if job_id not in new_worker_assignments_combine:
                    new_worker_assignments_combine[job_id] = {}
                new_worker_assignments_combine[job_id][cname] = new_worker_assignments[cname][job_id]
        
        new_worker_assignments = new_worker_assignments_combine
       

        # Verify the assignment.
        num_assignments = {}
        for job_id in new_worker_assignments:
            for cname in new_worker_assignments[job_id]:
                for worker_id in new_worker_assignments[job_id][cname]:
                    if worker_id not in num_assignments:
                        num_assignments[worker_id] = 0
                    num_assignments[worker_id] += 1
        for worker_id in num_assignments:
            if num_assignments[worker_id] != 1:
                raise RuntimeError('Worker {0} was assigned {1} times!'.format(worker_id, num_assignments[worker_id]))

        return new_worker_assignments

    def optimize(self, job_infos, nodes, prev_allocations):
        print("########################## Start ##################################")
        
        # determine the alloc
        # NOTE: allocs only contains the job which receives gpus
        _, _, allocs = self.solve(job_infos, self.iter_time_dict, self.valid_allocs)
        
        print('--- allocation: ---')
        for job, alloc in allocs.items():
            print(job, alloc)

        # map alloc to gpu_ids
        worker_assignments = self.schedule_jobs_on_workers(allocs)
        print('--- gpu assignment: ---')
        for job, ass in worker_assignments.items():
            print(job, ass)

        # update
        self.allocs = allocs
        self.worker_assignments = worker_assignments

        # convert to interface alloc format
        res = {}
        for job_id, worker_ids in self.worker_assignments.items():
            res[job_id] = {}
            for cname in worker_ids:
                ids = self.convert_worker_ids(cname, worker_ids[cname])
                res[job_id][cname] = ids

        print("--- converted_ids: ---")  
        print(res)   
        print("########################## End ##################################")            
        
        return res