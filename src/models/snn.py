import snntorch as snn
from snntorch import surrogate
import torch.nn as nn
import torch

class DynamicDeepSNN(nn.Module):
    def __init__(self, num_inputs, num_hidden=256, num_outputs=3):
        super().__init__()
        spike_grad = surrogate.fast_sigmoid()
        beta = 0.8 

        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc2 = nn.Linear(num_hidden, num_hidden)
        self.lif2 = snn.Leaky(beta=beta, spike_grad=spike_grad)
        self.fc3 = nn.Linear(num_hidden, num_outputs)
        self.lif3 = snn.Leaky(beta=beta, spike_grad=spike_grad)

        # 가중치 초기화: 죽은 뉴런 살리기
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        mem1, mem2, mem3 = self.lif1.init_leaky(), self.lif2.init_leaky(), self.lif3.init_leaky()
        spk_out_hist = []

        for step in range(x.size(0)):
            spk1, mem1 = self.lif1(self.fc1(x[step]), mem1)
            spk2, mem2 = self.lif2(self.fc2(spk1), mem2)
            spk3, mem3 = self.lif3(self.fc3(spk2), mem3)
            spk_out_hist.append(spk3)

        return torch.stack(spk_out_hist), mem3 #