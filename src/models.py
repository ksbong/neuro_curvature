import torch
import snntorch as snn
from snntorch import surrogate

class SimpleSNN(torch.nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs):
        super().__init__()
        self.fc1 = torch.nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())
        self.fc2 = torch.nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=0.9, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x):
        # x shape: (time_steps, batch, num_inputs)
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()
        
        spk2_rec = [] # 스파이크 기록용 리스트
        
        # 시간축을 따라 루프 실행
        for step in range(x.size(0)):
            cur1 = self.fc1(x[step])
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)
            spk2_rec.append(spk2)

        return torch.stack(spk2_rec), mem2