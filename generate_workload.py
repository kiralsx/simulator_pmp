import argparse
import numpy as np
import os
import pandas
import random
import pandas as pd

from datetime import datetime, timedelta

from applications_homo import APPLICATIONS


def generate_pmp(seed=0, is_itp_trace=False):
    if not is_itp_trace:
        trace_csv = os.path.join(os.path.dirname(__file__),
                             "traces", "philly.csv")
    else:
        trace_csv = os.path.join(os.path.dirname(__file__), 
                             "traces", "itp.csv")

    num_days = 7
    num_jobs = num_days * 20 * 24

    trace = pandas.read_csv(trace_csv, parse_dates=["timestamp"])
    trace = trace[trace.duration >= 60]
    trace = trace[trace.gpu_time < 1000 * 3600]

    start_tstamp = pd.Timestamp('2017-10-28')
    end_tstamp = start_tstamp + timedelta(hours=24*7)
    trace = trace[trace.timestamp > start_tstamp]
    trace = trace[trace.timestamp < end_tstamp]

    n_jobs = {}
    for i, row in trace.iterrows():
        if str(row['timestamp'].date()) not in n_jobs:
            n_jobs[str(row['timestamp'].date())] = 0
        n_jobs[str(row['timestamp'].date())] += 1
    n_jobs = dict(sorted(n_jobs.items(), key=lambda item: item[0]))
    for date, time in n_jobs.items():
        print(date, time)

    print(f"Sampling range: t = {start_tstamp} -> {end_tstamp}")
    trace.timestamp = trace.timestamp.apply(lambda x : x - trace.timestamp.min())
    print(f"Number of jobs in sampling range: {len(trace)}")
    print(f"First and last entries: {trace.iloc[0, -1]}")

    rng = random.Random(seed)
    rng2 = random.Random(seed + 3)
    rng3 = random.Random(seed + 4)
    sample = trace.sample(n=num_jobs, random_state=rng.randint(0, 1 << 32))

    records = []
    max_tstamp = sample.timestamp.max()
    print(f"Max timestamp: {max_tstamp}, type={type(max_tstamp)}")
    for row in sample.itertuples():
        rec = {"time": row.timestamp.total_seconds()}
        num_gpus = row.num_gpus
        if row.gpu_time < 1 * 3600:
            rec["application"] = rng.choice(["cifar10", "ncf"])
            # rec["application"] = rng.choice(["cifar10"])
        elif row.gpu_time < 10 * 3600:
            rec["application"] = "deepspeech2"
        elif row.gpu_time < 100 * 3600:
            rec["application"] = "yolov3"
            subset = trace[trace.duration <= 24 * 3600]
            subset = subset[subset.gpu_time >= 10 * 3600]
            subset = subset[subset.gpu_time < 100 * 3600]
            # num_gpus = rng3.choice(subset.num_gpus.to_list())
        else:
            rec["application"] = "imagenet"
            subset = trace[trace.duration <= 24 * 3600]
            subset = subset[subset.gpu_time >= 100 * 3600]
            # num_gpus = rng3.choice(subset.num_gpus.to_list())
        rec["num_replicas"], rec["batch_size"], _ = rng.choice(
            APPLICATIONS[rec["application"]].get_configurations(0.5, 0.8))
        if rec["application"] == "deepspeech2" and rng2.randint(0, 1):
            # Change half of the deepspeech2 jobs to bert jobs. Use a different
            # random number generator to avoid affecting the rest of the jobs.
            rec["application"] = "bert"
            rec["num_replicas"], rec["batch_size"], _ = rng2.choice(
                APPLICATIONS[rec["application"]].get_configurations(0.5, 0.8))
        #rec["num_replicas"] = num_gpus
        #rec["batch_size"] = APPLICATIONS[rec["application"]].init_batch_size * num_gpus
        records.append(rec)
    records.sort(key=lambda v: v["time"])
    for idx, rec in enumerate(records):
        rec["name"] = "{}-{}".format(rec["application"], idx)
    return pandas.DataFrame(records, columns=("name", "time", "application",
                                              "num_replicas", "batch_size"))

def generate(num_jobs, start=0, duration=24, seed=0, is_itp_trace=False):
    if not is_itp_trace:
        trace_csv = os.path.join(os.path.dirname(__file__),
                             "traces", "philly.csv")
    else:
        trace_csv = os.path.join(os.path.dirname(__file__), 
                             "traces", "itp.csv")

    num_days = 7
    num_jobs = num_days * 20 * 24

    trace = pandas.read_csv(trace_csv, parse_dates=["timestamp"])
    trace = trace[trace.duration >= 60]
    trace = trace[trace.gpu_time < 1000 * 3600]

    start_tstamp = pd.Timestamp('2017-10-28')
    end_tstamp = start_tstamp + timedelta(hours=24*7)
    trace = trace[trace.timestamp > start_tstamp]
    trace = trace[trace.timestamp < end_tstamp]

    n_jobs = {}
    for i, row in trace.iterrows():
        if str(row['timestamp'].date()) not in n_jobs:
            n_jobs[str(row['timestamp'].date())] = 0
        n_jobs[str(row['timestamp'].date())] += 1
    n_jobs = dict(sorted(n_jobs.items(), key=lambda item: item[0]))
    for date, time in n_jobs.items():
        print(date, time)

    print(f"Sampling range: t = {start_tstamp} -> {end_tstamp}")
    trace.timestamp = trace.timestamp.apply(lambda x : x - start_tstamp)
    print(f"Number of jobs in sampling range: {len(trace)}")
    print(f"First and last entries: {trace.iloc[0, -1]}")

    rng = random.Random(seed)
    rng2 = random.Random(seed + 3)
    rng3 = random.Random(seed + 4)
    sample = trace.sample(n=num_jobs, random_state=rng.randint(0, 1 << 32))

    records = []
    max_tstamp = sample.timestamp.max()
    print(f"Max timestamp: {max_tstamp}, type={type(max_tstamp)}")
    for row in sample.itertuples():
        rec = {"time": row.timestamp.total_seconds()}
        num_gpus = row.num_gpus
        if row.gpu_time < 1 * 3600:
            rec["application"] = rng.choice(["cifar10", "ncf"])
        elif row.gpu_time < 10 * 3600:
            rec["application"] = "deepspeech2"
        elif row.gpu_time < 100 * 3600:
            rec["application"] = "yolov3"
            subset = trace[trace.duration <= 24 * 3600]
            subset = subset[subset.gpu_time >= 10 * 3600]
            subset = subset[subset.gpu_time < 100 * 3600]
            # num_gpus = rng3.choice(subset.num_gpus.to_list())
        else:
            rec["application"] = "imagenet"
            subset = trace[trace.duration <= 24 * 3600]
            subset = subset[subset.gpu_time >= 100 * 3600]
            # num_gpus = rng3.choice(subset.num_gpus.to_list())
        rec["num_replicas"], rec["batch_size"], _ = rng.choice(
            APPLICATIONS[rec["application"]].get_configurations(0.5, 0.8))
        if rec["application"] == "deepspeech2" and rng2.randint(0, 1):
            # Change half of the deepspeech2 jobs to bert jobs. Use a different
            # random number generator to avoid affecting the rest of the jobs.
            rec["application"] = "bert"
            rec["num_replicas"], rec["batch_size"], _ = rng2.choice(
                APPLICATIONS[rec["application"]].get_configurations(0.5, 0.8))
        #rec["num_replicas"] = num_gpus
        #rec["batch_size"] = APPLICATIONS[rec["application"]].init_batch_size * num_gpus
        records.append(rec)
    exit()
    for idx, rec in enumerate(records):
        rec["name"] = "{}-{}".format(rec["application"], idx)
    return pandas.DataFrame(records, columns=("name", "time", "application",
                                              "num_replicas", "batch_size"))



    trace = pandas.read_csv(trace_csv, parse_dates=["timestamp"])
    trace = trace[trace.duration >= 60]
    trace = trace[trace.gpu_time < 1000 * 3600]
    print(f"Trace limits: {trace.timestamp[0]} -> {trace.timestamp[len(trace) - 1]}")
    start_tstamp = trace.timestamp.min() + timedelta(hours=start)
    end_tstamp = start_tstamp + timedelta(hours=duration)
    print(f"Sampling range: t = {start_tstamp} -> {end_tstamp}")
    trace = trace[trace.timestamp > start_tstamp]
    trace = trace[trace.timestamp < end_tstamp]
    trace.timestamp = trace.timestamp.apply(lambda x : x - start_tstamp)
    print(f"Number of jobs in sampling range: {len(trace)}")
    print(f"First and last entries: {trace.iloc[0, -1]}")

    exit()

    # trace.timestamp -= timedelta(hours=start)
    rng = random.Random(seed)
    rng2 = random.Random(seed + 3)
    rng3 = random.Random(seed + 4)
    sample = trace.sample(n=num_jobs, random_state=rng.randint(0, 1 << 32))

    records = []
    max_tstamp = sample.timestamp.max()
    print(f"Max timestamp: {max_tstamp}, type={type(max_tstamp)}")
    for row in sample.itertuples():
        rec = {"time": row.timestamp.total_seconds()}
        num_gpus = row.num_gpus
        if row.gpu_time < 1 * 3600:
            rec["application"] = rng.choice(["cifar10", "ncf"])
        elif row.gpu_time < 10 * 3600:
            rec["application"] = "deepspeech2"
        elif row.gpu_time < 100 * 3600:
            rec["application"] = "yolov3"
            subset = trace[trace.duration <= 24 * 3600]
            subset = subset[subset.gpu_time >= 10 * 3600]
            subset = subset[subset.gpu_time < 100 * 3600]
            # num_gpus = rng3.choice(subset.num_gpus.to_list())
        else:
            rec["application"] = "imagenet"
            subset = trace[trace.duration <= 24 * 3600]
            subset = subset[subset.gpu_time >= 100 * 3600]
            # num_gpus = rng3.choice(subset.num_gpus.to_list())
        rec["num_replicas"], rec["batch_size"], _ = rng.choice(
            APPLICATIONS[rec["application"]].get_configurations(0.5, 0.8))
        if rec["application"] == "deepspeech2" and rng2.randint(0, 1):
            # Change half of the deepspeech2 jobs to bert jobs. Use a different
            # random number generator to avoid affecting the rest of the jobs.
            rec["application"] = "bert"
            rec["num_replicas"], rec["batch_size"], _ = rng2.choice(
                APPLICATIONS[rec["application"]].get_configurations(0.5, 0.8))
        #rec["num_replicas"] = num_gpus
        #rec["batch_size"] = APPLICATIONS[rec["application"]].init_batch_size * num_gpus
        records.append(rec)
    records.sort(key=lambda v: v["time"])
    print(np.max([r['time'] for r in records])/3600)
    print(np.min([r['time'] for r in records])/3600)
    print([r['time'] for r in records])
    exit()
    for idx, rec in enumerate(records):
        rec["name"] = "{}-{}".format(rec["application"], idx)
    return pandas.DataFrame(records, columns=("name", "time", "application",
                                              "num_replicas", "batch_size"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--start", type=int, default=2,
                        help="starting hour")
    parser.add_argument("-d", "--duration", type=int, default=8,
                        help="total number of workload hours")
    parser.add_argument("-n", "--num-jobs", type=int, default=160,
                        help="total number of jobs")
    parser.add_argument("-o", "--output", type=str,
                        help="path to output the workload")
    parser.add_argument("--seed", type=int, default=0,
                        help="random seed")
    parser.add_argument("--itp_traces", action='store_true', default=False, help="use itp traces")
    args = parser.parse_args()
    # workload = generate(args.num_jobs, start=args.start,
    #                     duration=args.duration, seed=args.seed, is_itp_trace=args.itp_traces)
    workload = generate_pmp()
    csv = workload.set_index("name").to_csv(args.output)
    if csv:
        print(csv)
    print(workload.groupby(["application", "num_replicas", "batch_size"])
          .size().reset_index(name="count"))
