# Target Dataset -> CIFAR10
import torch
import torch.nn as nn
import torch.nn.functional as F


# 간단한 base encoder -> Resnet으로 바꾸는 것을 추천.
class Base_Encoder(nn.Module):
    def __init__(self, ):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(3, 32, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.ReLU()
        )
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 128)

    def forward(self, x):
        x = self.net(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class MoCo(nn.Module):
    def __init__(self, queue_length=32768, m=0.999,
                 temperature=0.7):  # queue length 는 batch size로 나눴을 때 나눠 떨어져야 한다. -> 2의 제곱승이면 좋을듯
        super().__init__()
        self.q = Base_Encoder()
        self.k = Base_Encoder()
        self.m = m
        self.queue_length = queue_length  # data num
        self.T = temperature

        # initialization
        for q_param, k_param in zip(self.q.parameters(), self.k.parameters()):
            k_param.data.copy_(q_param.data)
            k_param.requires_grad = False

        self.register_buffer("queue", torch.randn(128, self.queue_length))  # 초반 노이즈로 초기화
        self.queue = F.normalize(self.queue, dim=0)
        self.register_buffer("ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def move_average_encoder(self):
        for q_param, k_param in zip(self.q.parameters(), self.k.parameters()):
            k_param.data.copy_(q_param.data * self.m + k_param.data * (1 - self.m))
            k_param.requires_grad = False

    @torch.no_grad()
    def dequeue_enqueue(self, keys):  # keys size -> bs, dim
        bs = keys.size(0)
        ptr = int(self.ptr)
        self.queue[:, ptr: ptr + bs] = keys.T

        ptr = (ptr + bs) % self.queue_length
        self.ptr[0] = ptr

    def forward(self, x, others):

        q = self.q(x)
        q = F.normalize(q, dim=1)

        with torch.no_grad():
            self.move_average_encoder()

            k = self.k(others)
            k = F.normalize(k)

        # 여기에서의 k는 positive
        pos = torch.mm(q, k.T)  # n by n

        # queue에 있는 친구들은 negative
        neg = torch.mm(q, self.queue.detach().clone())  # queue 는 dim * queue Length 결과 : n by queue length

        logits = torch.cat([pos, neg], dim=1)

        # apply temperature
        logits /= self.T

        # labels: positive key indicators
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(x.device)

        self.dequeue_enqueue(k)

        return logits, labels

