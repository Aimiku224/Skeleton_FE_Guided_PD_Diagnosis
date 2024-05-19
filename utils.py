import os
import random
import numpy as np
import torch

def seed_model(seed=140224, gpu_id=0):
    torch.cuda.set_device(gpu_id)
    seed = int(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
#    torch.use_deterministic_algorithms(True)





