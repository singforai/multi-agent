import numpy as np
import torch

old_action_log_probs_batch = torch.tensor((5000,3, 1))
print(old_action_log_probs_batch.shape)
old_action_log_probs_batch.reshape(-1, 3, old_action_log_probs_batch.shape[-1]).sum(dim=(1, -1), keepdim=True)
print(old_action_log_probs_batch.shape)