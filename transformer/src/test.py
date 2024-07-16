import numpy as np
from statistics import mean

base_dir = '/cephyr/users/zesenw/Alvis/work/nips24-accum-adam/transformer/log/2485825'

for i in range(16):
    data = np.load(f'{base_dir}/time_00_{i:02d}.npy', allow_pickle=True).item()
    print(f"{mean(data['forward']):.4f} {mean(data['iter']):.4f}")
    print(len(data['forward']), len(data['iter']))
