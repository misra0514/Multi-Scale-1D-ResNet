import numpy as np
import time
from visdom import Visdom

# 用于输出损失等等信息的函数
# visdom 是一个用于绘图的库函数（类matplotlib），但是会放在服务器上
# TODO: 有时候无法链接。 python -m visdom.server
# 之后记得ssh -L 8097:127.0.0.1:8097 liangyishuang@10.0.87.17 一下。
class line(object):
    def __init__(self, title, port=8080):
        self.vis = Visdom(port=port)
        self.windows = {}
        self.title = title

    def register_line(self, name, xlabel, ylabel): # new a plot
        win = self.vis.line(
            X=np.zeros(3),
            Y=np.zeros(3),
            opts=dict(title=self.title, markersize=5, xlabel=xlabel, ylabel=ylabel)
        )
        self.windows[name] = win

    def update_line(self, name, x, *args):
        axis = ()
        for _ in range(len(args)):
            axis += (x+1,)

        # vis.line : 绘制曲线图
        self.vis.line(
            X=np.array([axis]),
            Y=np.array([args]),
            win=self.windows[name],
            update='append'
        )



if __name__ == '__main__':
    line = line('loss', port=8097)
    line.register_line('loss', 'iter', 'loss') # 初始化

    for x in range(5):
        line.update_line('loss', x, 10, 20, 30)
