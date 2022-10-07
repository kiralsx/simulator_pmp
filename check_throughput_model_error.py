from applications import APPLICATIONS
from goodput import _predict_accum_time, _predict_log_optim_time, _predict_network_time, fit_perf_params
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
import argparse

def get_performance_profiles(cluster_name, app_name, selector_bsize=None):
  app = APPLICATIONS[cluster_name][app_name]
  df = app.scalability
  num_nodes = df.num_nodes.to_numpy()
  num_replicas = df.num_replicas.to_numpy()
  local_bsz = df.local_bsz.to_numpy()
  sync_time = df.sync_time.to_numpy()
  step_time = df.step_time.to_numpy()
  # include placements.csv too
  df = app.placements
  def sum_digits(val):
    return sum([int(v) for v in str(val)])
  def num_digits(val):
    return len(str(val))
  p_num_nodes = np.vectorize(num_digits)(df.placement.to_numpy())
  p_num_replicas = np.vectorize(sum_digits)(df.placement.to_numpy())
  p_step_time = df.step_time.to_numpy()
  p_local_bsz = df.local_bsz.to_numpy()
  p_sync_time = df.sync_time.to_numpy()
  
  # append placement profiles
  if True:
    num_nodes = np.hstack((num_nodes, p_num_nodes))
    num_replicas = np.hstack((num_replicas, p_num_replicas))
    local_bsz = np.hstack((local_bsz, p_local_bsz))
    step_time = np.hstack((step_time, p_step_time))
    sync_time = np.hstack((sync_time, p_sync_time))

  if selector_bsize is not None:
    selector_vec = (local_bsz == selector_bsize)
    num_nodes = num_nodes[selector_vec]
    num_replicas = num_replicas[selector_vec]
    local_bsz = local_bsz[selector_vec]
    sync_time = sync_time[selector_vec]
    step_time = step_time[selector_vec]
  return num_nodes, num_replicas, local_bsz, step_time, sync_time

def build_xput_model(num_nodes, num_replicas, atomic_bsz, step_time, sync_time):
  compute_time = step_time - sync_time
  perf_params = fit_perf_params(num_nodes, num_replicas, atomic_bsz, compute_time, step_time)
  return perf_params

def predict_xput_model(perf_params, num_nodes, num_replicas, atomic_bsz):
  pred_t_grad = _predict_accum_time(perf_params, atomic_bsz)
  pred_t_sync = _predict_network_time(perf_params, num_nodes, num_replicas)
  pred_log_t_iter = _predict_log_optim_time(perf_params, pred_t_grad, pred_t_sync)
  pred_t_iter = np.exp(pred_log_t_iter)
  return pred_t_iter, pred_t_sync

def lookup_placement(num_nodes, num_replicas, local_bsz):
  num_replicas_per_node = int(num_replicas / num_nodes)
  placement_id = int("".join([str(num_replicas_per_node)]*num_nodes))
  entry = pl_df.query(f'placement == {placement_id} & local_bsz == {atomic_bsz}')
  # print(f"Looking up placement: {placement_id}, entry={entry}")
  if entry.empty:
    return None
  else:
    return entry.step_time.item(), entry.sync_time.item()

def lookup_scalability(num_nodes, num_replicas, local_bsz):
  entry = sc_df.query(f'num_nodes == {num_nodes} & num_replicas == {num_replicas} & local_bsz == {local_bsz}')
  if entry.empty:
    return None
  else:
    return entry.step_time.item(), entry.sync_time.item()

# process args
parser = argparse.ArgumentParser()
parser.add_argument("--cluster", type=str, help="cluster name")
parser.add_argument("--app", type=str, help="app name")
parser.add_argument("--bsize", type=int, help="batch size")
args = parser.parse_args()
cluster_name, app_name = args.cluster, args.app
atomic_bsz = args.bsize

app = APPLICATIONS[cluster_name][app_name]
sc_df = app.scalability
pl_df = app.placements

# build throughput model from performance profiles
num_nodes, num_replicas, local_bsz, step_time, sync_time = get_performance_profiles(cluster_name, app_name)
app_perf_params = build_xput_model(num_nodes, num_replicas, local_bsz, step_time, sync_time)

# predict sync_time, xput for a fixed bsize
pred_selector = local_bsz == atomic_bsz
pred_num_nodes, pred_num_replicas = num_nodes[pred_selector], num_replicas[pred_selector]
step_time, local_bsz, sync_time = step_time[pred_selector], local_bsz[pred_selector], sync_time[pred_selector]
pred_step_time, pred_sync_time = predict_xput_model(app_perf_params, pred_num_nodes, pred_num_replicas, local_bsz)
pred_xput, xput = local_bsz / pred_step_time, local_bsz / step_time

def compute_relative_error(x, y):
  return np.abs(x - y) / y
step_error = compute_relative_error(pred_step_time,step_time)
xput_error = compute_relative_error(pred_xput,xput)

print(f"\tstep_time prediction error: max={np.max(step_error)*100}%, avg={np.mean(step_error)*100}%")
print(f"\txput prediction error: max={np.max(xput_error)*100}%, avg={np.mean(xput_error)*100}%")
