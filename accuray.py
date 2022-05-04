#%%
import tensorboard as tb
import numpy as np
import os
#%%
model = 'dgm'
dir = '/Users/anseunghwan/Documents/GitHub/semi/{}/logs/cifar10_4000'.format(model)
file_list = [x for x in os.listdir(dir) if x != '.DS_Store']
#%%
import traceback
import pandas as pd
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# Extraction function
def tflog2pandas(path):
    runlog_data = pd.DataFrame({"metric": [], "value": [], "step": []})
    try:
        event_acc = EventAccumulator(path)
        event_acc.Reload()
        tags = event_acc.Tags()["tensors"]
        tag = 'accuracy'
        for tag in tags:
            event_list = event_acc.Tensors(tag)
            values = list(map(lambda x: x.value, event_list))
            step = list(map(lambda x: x.step, event_list))
            r = {"metric": [tag] * len(step), "value": values, "step": step}
            r = pd.DataFrame(r)
            runlog_data = pd.concat([runlog_data, r])
    # Dirty catch of DataLossError
    except Exception:
        print("Event file possibly corrupt: {}".format(path))
        traceback.print_exc()
    return runlog_data
#%%
path = dir + '/{}/test'.format(file_list[0])
df = tflog2pandas(path)
#df=df[(df.metric != 'params/lr')&(df.metric != 'params/mm')&(df.metric != 'train/loss')] #delete the mentioned rows
df.to_csv("output.csv")
#%%
experiment = tb.data.experimental.ExperimentFromDev(path)
experiment.get_scalars()

#%%
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
event_acc = EventAccumulator(path)
event_acc.Reload()
# Show all tags in the log file
print(event_acc.Tags())

w_times, step_nums, vals = zip(*event_acc.Scalars('tensors'))
#%%