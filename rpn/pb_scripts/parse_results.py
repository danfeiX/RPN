import json
import sys
import numpy as np

rfile = json.load(open(sys.argv[1]))
print('problem: %s' % rfile['args']['problem'])
avg_scc_rate = np.mean(rfile['success'])
completion = np.array(rfile['completion']).astype(np.float64)
avg_com_rate = np.mean(completion[:, 0] / completion[:, 1])
print('avg_success_rate: %f' % avg_scc_rate)
print('avg_completion_rate: %f' % avg_com_rate)