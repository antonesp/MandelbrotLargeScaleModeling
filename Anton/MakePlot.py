import matplotlib.pyplot as plt
import numpy as np

c2 = [39.195029497146606,
40.60849857330322,
36.82233715057373,
36.60610055923462,
49.05009627342224,
]
c4 = [14.569204330444336,
14.167542457580566,
14.873608112335205,
15.309185981750488,
14.811164140701294]
c8 = [6.37129545211792,
6.37081503868103,
6.547433614730835,
6.191807270050049,
6.509135484695435,
]
c16 = [3.1367506980895996,
3.0763838291168213,
3.0404539108276367,
3.13086199760437,
3.120816707611084,
]
cores = np.array([2, 4, 8, 16])
measurements = [c2, c4, c8, c16]
means = [np.mean(values) for values in measurements]

plt.plot(cores, means, marker="o")
plt.xlabel("Number of cores")
plt.ylabel("Time (sec)")
plt.title("Mean time by number of cores")
plt.grid(True)
plt.tight_layout()
plt.savefig("mean_measurement_by_cores.png")


