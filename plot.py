import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

data = pd.DataFrame()
data = data.append({'type': 'homoengeneous', 'workloads': 'philly-0.5', 'avg jct': 0.39}, ignore_index=True)
data = data.append({'type': 'heterogeneous', 'workloads': 'philly-0.5', 'avg jct': 0.36}, ignore_index=True)
data = data.append({'type': 'homoengeneous', 'workloads': 'philly', 'avg jct': 0.77}, ignore_index=True)
data = data.append({'type': 'heterogeneous', 'workloads': 'philly', 'avg jct': 0.65}, ignore_index=True)
data = data.append({'type': 'homoengeneous', 'workloads': 'philly-no-ncf', 'avg jct': 0.93}, ignore_index=True)
data = data.append({'type': 'heterogeneous', 'workloads': 'philly-no-ncf', 'avg jct': 0.78}, ignore_index=True)



fig, ax = plt.subplots()
splot=sns.barplot(x="workloads", y="avg jct", hue="type", data=data, orient='v')
ax.set_xlabel("Workloads", fontsize=15)
ax.set_ylabel(f"Avg jct", fontsize=15)
ax.set_ylim(ymin=0)
# ax.set_xlim(xmin=0)
plt.title(f'24 RTX + 16 A100')
plt.legend(loc="best")
plt.tight_layout()
plt.savefig('./plots/scheduling_pmp.png')