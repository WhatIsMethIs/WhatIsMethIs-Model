from torch import nn


class PillModel(nn.Module):
    def __init__(self, config):
        super(PillModel, self).__init__()
        """
        ClassNum : class number
        """

        self.m_ClassNum = int(config["class_num"])

        channel1 = 16
        channel2 = 32
        channel3 = 64
        conv1Size = 3
        conv2Size = 3
        poolSize = 2

        self.m_Conv1 = nn.Conv2d(in_channels=3, out_channels=channel1, kernel_size=conv1Size, padding=1)
        self.m_Pool1 = nn.MaxPool2d(poolSize, poolSize)
        self.m_Conv2 = nn.Conv2d(in_channels=channel1, out_channels=channel2, kernel_size=conv2Size, padding=1)
        self.m_Pool2 = nn.MaxPool2d(poolSize, poolSize)
        self.m_Conv3 = nn.Conv2d(in_channels=channel2, out_channels=channel3, kernel_size=conv2Size, padding=1)
        self.m_Pool3 = nn.MaxPool2d(poolSize, poolSize)

        self.m_Linear4 = nn.Linear(40000, 256)
        self.m_Drop4 = nn.Dropout2d(0.5)

        self.m_Linear5 = nn.Linear(256, self.m_ClassNum)
        self.m_Relu = nn.ReLU()

    def forward(self, x):
        x = self.m_Relu(self.m_Conv1(x))
        x = self.m_Pool1(x)

        x = self.m_Relu(self.m_Conv2(x))
        x = self.m_Pool2(x)

        x = self.m_Relu(self.m_Conv3(x))
        x = self.m_Pool3(x)

        x = x.view(x.shape[0], -1)
        x = self.m_Relu(self.m_Linear4(x))
        x = self.m_Drop4(x)

        x = self.m_Linear5(x)
        return x
