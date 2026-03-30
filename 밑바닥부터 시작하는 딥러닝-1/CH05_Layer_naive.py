class MulLayer:         # 곱셈 계층
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y
        
        return out
    
    def backward(self, dout):
        dx = dout * self.y      # x 와 y를 바꾼다.
        dy = dout * self.x      # y와 x를 바꾼다.
                                # dout = 상류에서 넘어온 미분
        return dx, dy
    

class AddLayer:
    def __init__(self):
        pass                # 초기화 필요 없음 : 그냥 상류의 미분값(dout *1)을 보내주면 됨

    def forward(self, x, y):
        out = x + y
        return out

    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy