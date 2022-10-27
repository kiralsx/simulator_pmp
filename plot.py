import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import json
import numpy as np

# data = pd.DataFrame()
# data = data.append({'type': 'homoengeneous', 'workloads': 'philly-0.5', 'avg jct': 0.39}, ignore_index=True)
# data = data.append({'type': 'heterogeneous', 'workloads': 'philly-0.5', 'avg jct': 0.36}, ignore_index=True)
# data = data.append({'type': 'homoengeneous', 'workloads': 'philly', 'avg jct': 0.77}, ignore_index=True)
# data = data.append({'type': 'heterogeneous', 'workloads': 'philly', 'avg jct': 0.65}, ignore_index=True)
# data = data.append({'type': 'homoengeneous', 'workloads': 'philly-no-ncf', 'avg jct': 0.93}, ignore_index=True)
# data = data.append({'type': 'heterogeneous', 'workloads': 'philly-no-ncf', 'avg jct': 0.78}, ignore_index=True)


# data = pd.DataFrame()
# data = data.append({'type': 'homoengeneous (1 dgx, 20 rtx)', 'workloads': 'philly-0.5', 'avg jct': 3952/3600}, ignore_index=True)
# data = data.append({'type': 'heterogeneous (1 dgx, 20 rtx)', 'workloads': 'philly-0.5', 'avg jct': 3018/3600}, ignore_index=True)
# data = data.append({'type': 'homoengeneous (1 dgx, 20 rtx)', 'workloads': 'philly', 'avg jct': 8786/3600}, ignore_index=True)
# data = data.append({'type': 'heterogeneous (1 dgx, 20 rtx)', 'workloads': 'philly', 'avg jct': 7502/3600}, ignore_index=True)
# data = data.append({'type': 'homoengeneous (1 dgx, 20 rtx)', 'workloads': 'philly-no-ncf', 'avg jct': 8994/3600}, ignore_index=True)
# data = data.append({'type': 'heterogeneous (1 dgx, 20 rtx)', 'workloads': 'philly-no-ncf', 'avg jct': 7792/3600}, ignore_index=True)


# # 2 + 10
# # data = pd.DataFrame()
# data = data.append({'type': 'homoengeneous (2 dgx, 10 rtx)', 'workloads': 'philly-0.5', 'avg jct': 3014/3600}, ignore_index=True)
# data = data.append({'type': 'heterogeneous (2 dgx, 10 rtx)', 'workloads': 'philly-0.5', 'avg jct': 2769/3600}, ignore_index=True)
# data = data.append({'type': 'homoengeneous (2 dgx, 10 rtx)', 'workloads': 'philly', 'avg jct': 8271/3600}, ignore_index=True)
# data = data.append({'type': 'heterogeneous (2 dgx, 10 rtx)', 'workloads': 'philly', 'avg jct': 7495/3600}, ignore_index=True)
# data = data.append({'type': 'homoengeneous (2 dgx, 10 rtx)', 'workloads': 'philly-no-ncf', 'avg jct': 8537/3600}, ignore_index=True)
# data = data.append({'type': 'heterogeneous (2 dgx, 10 rtx)', 'workloads': 'philly-no-ncf', 'avg jct': 7778/3600}, ignore_index=True)



# fig, ax = plt.subplots()
# splot=sns.barplot(x="workloads", y="avg jct", hue="type", data=data, orient='v')
# ax.set_xlabel("Workloads", fontsize=15)
# ax.set_ylabel(f"Avg jct", fontsize=15)
# ax.set_ylim(ymin=0, ymax=2.5)
# # ax.set_xlim(xmin=0)
# # plt.title(f'2 DGX + 10 RTX')
# plt.legend(loc="best")
# plt.tight_layout()
# plt.savefig('./plots/different_cluster.png')



# exit()

# read single summary
def read_summary(path):
    summary = {}
    try:
        with open(path, 'r') as f:
            summary = json.load(f)
    except:
        print(f"ERROR: failed to read summary at {path}")
        raise
    return summary

summary_path = './results/workloads-0.5/homo/summary.json'


def get_jct(summary_path):
    summary = read_summary(summary_path)
    job_jct = {}
    for _, workload_jcts in summary['jcts'].items():
        for jobname, jct in workload_jcts.items():
            app_name = jobname.split('-')[0]
            if app_name not in job_jct:
                job_jct[app_name] = []
            job_jct[app_name].append(jct)

    return {k: np.mean(v) for k,v in job_jct.items()}

# cluster_size = ['2dgx_10rtx', '1dgx_20rtx']
# config_type = ['homogeneous', 'heterogeneous']
# workloads = {'workloads-0.5': 'philly-0.5', 'workloads': 'philly', 'no_ncf_workloads': 'philly-no-ncf'}

# for cluster in cluster_size:
#     for workload in workloads:
#         data = pd.DataFrame()
#         for type in config_type:
#             summary_path = f'./results/{cluster}/{workload}/{type[:4]}/summary.json'
#             jcts = get_jct(summary_path)
#             for jobname, jct in jcts.items():
#                 data = data.append({'type': type, 'job': jobname, 'jct': jct}, ignore_index=True)

#         fig, ax = plt.subplots()
#         splot=sns.barplot(x="job", y="jct", hue="type", data=data, orient='v')
#         ax.set_xlabel("Job Type", fontsize=15)
#         ax.set_ylabel(f"Avg jct", fontsize=15)
#         ax.set_ylim(ymin=0)
#         # ax.set_xlim(xmin=0)
#         plt.title(f'{workloads[workload]}')
#         plt.legend(loc="best")
#         plt.tight_layout()
#         plt.savefig(f'./plots/{workload}.png')

""""""
# for job in ['cifar10', 'deepspeech2', 'ncf', 'yolov3', 'bert', 'imagenet']:
#     data = pd.DataFrame()
#     for cluster in cluster_size:
#         for workload in workloads:
#             for type in config_type:
#                 summary_path = f'./results/{cluster}/{workload}/{type[:4]}/summary.json'
#                 jcts = get_jct(summary_path)
#                 if job not in jcts:
#                     continue
#                 jct = jcts[job]
#                 data = data.append({'type': cluster + ' ' + type, 'workload':workload, 'jct': jct}, ignore_index=True)

#     fig, ax = plt.subplots()
#     splot=sns.barplot(x="workload", y="jct", hue="type", data=data, orient='v')
#     ax.set_xlabel("workload", fontsize=15)
#     ax.set_ylabel(f"Avg jct", fontsize=15)
#     ax.set_ylim(ymin=0)
#     # ax.set_xlim(xmin=0)
#     plt.title(f'{job}')
#     plt.legend(loc="best")
#     plt.tight_layout()
#     plt.savefig(f'./plots/{job}.png')


plt.clf()
label_added = False
barwidth=0.2

def plot_points(points, offset, display_str):
  cur_sum = np.zeros_like(points[0])
  global label_added
  for i, cur_points in enumerate(points):
    print(cur_sum)
    cluster = cluster_ordering[i]
    label = cluster_names[cluster] if not label_added else None
    b = plt.bar(Xs + offset, cur_points, label=label, color=cluster_colors[cluster], 
                bottom=cur_sum, width=barwidth)
    cur_sum += cur_points
    
  for x, y in zip(Xs + offset - barwidth / 2 + 0.02, 0.05 + np.zeros_like(Xs)):
    plt.gca().text(x, y, display_str, fontsize=12, fontweight='bold', color='w')
  label_added = True

offsets = [-0.25, 0, 0.25]
plot_pts = [haware_points, pollux_points, gavel_points]
display_strs = ['A', 'P', 'G']
for plot_pt, offset, display_str in zip(plot_pts, offsets, display_strs):
  plot_points(plot_pt, offset, display_str)

plt.axhline(y=1.0, color='k', linewidth=0.3, linestyle='--')
for i in range(len(Xs) - 1):
  plt.axvline(x=Xs[i] + 0.5, color='k', linewidth=1)
plt.xticks(ticks=Xs, labels=[model_names[x] for x in model_ordering], rotation=0, fontsize=12)
plt.legend(fontsize=12)
plt.yticks(fontsize=12)
plt.gcf().set_size_inches(6, 2.25)
plt.gcf().set_tight_layout(tight=1)
plt.gcf().set_dpi(300)
plt.ylabel('norm. GPU hrs', fontsize=12)
plt.savefig('/tmp/norm_gpu_hrs.pdf')