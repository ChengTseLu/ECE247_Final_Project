
DROPOUT=0.5

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        # Layer 1
        self.conv11 = nn.Conv2d(1, 29, (1, 4))
        self.conv12 = nn.Conv2d(29, 29, (22, 1))
        self.batchnorm1 = nn.BatchNorm2d(29)
        
        # Layer 2
        self.conv2 = nn.Conv2d(29, 49, (1, 6))
        self.batchnorm2 = nn.BatchNorm2d(49)
        
        # Layer 3
        self.conv3 = nn.Conv2d(49, 104, (1, 5))
        self.batchnorm3 = nn.BatchNorm2d(104)

        # Layer 4
        # self.conv4 = nn.Conv2d(104, 197, (1, 4))
        # self.batchnorm4 = nn.BatchNorm2d(197)

        
        # FC Layer
        self.fc = nn.Linear(104 * 58, 4)

        self.pool = nn.MaxPool2d((1,2))
        
        self.dp = nn.Dropout(p=DROPOUT)

    def forward(self, x):
        # Layer 1
        # reshape x: (B, 22, 1000) -> (B, 1, 22, 1000), B,C,H,W
        x = x.view(-1, 1, 22, 500)

        #x = torch.transpose(x, 1, 2)
        x = self.conv12(self.conv11(x))
        x = self.batchnorm1(x)
        x = F.elu(x)
        x = self.pool(x)
        x = self.dp(x)
        
        # Layer 2
        x = self.conv2(x)
        x = self.batchnorm2(x)
        x = F.elu(x)
        x = self.pool(x)
        x = self.dp(x)
        
        # Layer 3
        x = self.conv3(x)
        x = self.batchnorm3(x)
        x = F.elu(x)
        x = self.pool(x)
        x = self.dp(x)
        
        # Layer 4
        # x = self.conv4(x)
        # x = self.batchnorm4(x)
        # x = F.elu(x)
        # x = self.pool(x)
        # x = self.dp(x)
        # print(x.shape)
        x = x.reshape(-1, 104 * 58)
        x = self.fc(x)

        return x


weight_decay = 1e-1 
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4, weight_decay=weight_decay)