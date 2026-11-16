import os
import subprocess
import sys

import pytest


@pytest.mark.parametrize('workspace', [None, ':16:8'])
def test_deterministic_backward_in_fresh_process(workspace):
    # Isolate CUDA initialization and environment state from other tests.
    env = os.environ.copy()
    env.pop('CUBLAS_WORKSPACE_CONFIG', None)
    if workspace is not None:
        env['CUBLAS_WORKSPACE_CONFIG'] = workspace
    script = '''
import os
import sys
import torch
from minimal_recap.config import seed_all

seed_all(7)
assert os.environ['CUBLAS_WORKSPACE_CONFIG'] == sys.argv[1]
assert torch.are_deterministic_algorithms_enabled()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def backward():
    seed_all(7)
    model = torch.nn.Linear(4, 3).to(device)
    x = torch.randn(8, 4, device=device, requires_grad=True)
    model(x).square().mean().backward()
    return model.weight.grad.clone(), x.grad.clone()

first, second = backward(), backward()
for a, b in zip(first, second):
    assert torch.isfinite(a).all()
    torch.testing.assert_close(a, b, rtol=0, atol=0)
'''
    subprocess.run(
        [sys.executable, '-c', script, workspace or ':4096:8'],
        env=env, check=True, capture_output=True, text=True,
    )
