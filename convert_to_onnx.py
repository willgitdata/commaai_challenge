import torch
from phasic_policy import PhasicPolicyNetwork
import os

os.makedirs('models', exist_ok=True)

model = PhasicPolicyNetwork(state_dim=3, action_dim=1, hidden_dim=64)
checkpoint = torch.load('/Users/billhumphrey/Documents/GitHub/controls_challenge/models/phasic_policy_best.pt', weights_only=True)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

dummy_state = torch.randn(1, 3, dtype=torch.float32)
dummy_action = torch.zeros(1, 1, dtype=torch.float32)

torch.onnx.export(model,
                 (dummy_state, dummy_action),
                 "models/phasic_policy_best.onnx",
                 export_params=True,
                 opset_version=11,
                 do_constant_folding=True,
                 input_names=['state', 'action'],
                 output_names=['output'])
