from . import BaseController
import numpy as np
import onnxruntime as ort

class Controller(BaseController):
    """
    A neural network controller using ONNX runtime
    """
    def __init__(self):
        self.ort_session = None
        self.prev_action = 0.0
        self.model_initialized = False
        
    def update(self, target_lataccel, current_lataccel, state, future_plan):
        # Initialize ONNX session if not already done
        if not self.model_initialized:
            self.ort_session = ort.InferenceSession(
                "./models/phasic_policy_best.onnx",  # Direct path to ONNX model
                providers=['CPUExecutionProvider']
            )
            self.model_initialized = True
            
        # Prepare inputs for the model
        state_input = np.array(state, dtype=np.float32).reshape(1, -1)
        action_input = np.array([[self.prev_action]], dtype=np.float32)
        
        # Run model inference
        outputs = self.ort_session.run(
            None, 
            {
                'state': state_input,
                'action': action_input
            }
        )
        
        # Update previous action and return current action
        action = float(outputs[0])
        self.prev_action = action
        return action

