from livn.system import make
from livn.io import MEA
from datasets import load_dataset

system_name = "S1"

dataset = load_dataset("livn-org/livn", name=system_name)

sample = dataset["train"][200]
it = sample["trial_it"][0]
t = sample["trial_t"][0]

# use a multi-electrode array to 'observe' the data
system = make(system_name)
mea = MEA.from_directory(system.uri)

# cit, ct = mea.channel_recording(system.neuron_coordinates, it, t)


# print("Neuron ID: ", it)
# print("Time: ", t)

# t_end = sample["t_end"]
# print("t_end: ", t_end)
print("Spike times: ", len(t))