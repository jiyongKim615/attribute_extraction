import torch as t
import time


class BasicModule(t.nn.Module):
    """
    save와 load 방식 제공
    """

    def __init__(self):
        super(BasicModule,self).__init__()
        self.model_name=str(type(self))# 默认名字

    def load(self, path):
        """
        지정된 경로가 있는 모델 로드
        """
        self.load_state_dict(t.load(path))

    def save(self, name=None):
        """
        모델 저장 ("모델명 + 시간"으로 파일명 사용)
        """
        if name is None:
            prefix = './checkpoints/' + self.model_name + '_'
            name = time.strftime(prefix + '%m%d_%H:%M:%S.pth')
        t.save(self.state_dict(), name)
        return name

    def get_optimizer(self, lr, weight_decay):
        return t.optim.Adam(self.parameters(), lr=lr, weight_decay=weight_decay)


class Flat(t.nn.Module):
    """
    입력을 (batch_size, dim_length)로 변형
    """

    def __init__(self):
        super(Flat, self).__init__()
        #self.size = size

    def forward(self, x):
        return x.view(x.size(0), -1)