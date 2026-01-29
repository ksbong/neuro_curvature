import torch
import snntorch as snn
from snntorch import surrogate

class GeometricSNN(torch.nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, beta=0.9):
        super().__init__()
        # 스파이크 기반 그래디언트 전달을 위해 surrogate gradient 사용
        spike_grad = surrogate.fast_sigmoid()
        
        self.fc1 = torch.nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = torch.nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)

    def forward(self, x):
        # x shape: (time_steps, batch, num_inputs)
        mem1 = self.lif1.init_leaky()
        # 이 부분! .index_copy 를 삭제해야 해
        mem2 = self.lif2.init_leaky() 
        
        spk2_rec = [] 

        for step in range(x.size(0)):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)

        return torch.stack(spk2_rec), mem2