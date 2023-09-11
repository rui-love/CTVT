import time
import numpy as np
from tqdm import tqdm

data = np.zeros(100)
try:
    for i in tqdm(range(100)):
        data[i] = i
        time.sleep(0.1)
except KeyboardInterrupt:
    print("KeyboardInterrupt")
    np.savetxt("data.csv", data, delimiter=",", fmt="%d")
else:
    print("No exception")
    np.savetxt("data.csv", data, delimiter=",", fmt="%d")
