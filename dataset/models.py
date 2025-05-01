import torch  
import torch.nn as nn  
import torch.nn.functional as F  

# Helper class for basic conv layers  
class FacenetLayer(nn.Module):  
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0):  
        super(FacenetLayer, self).__init__()  
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)  
        self.bn = nn.BatchNorm2d(out_planes, eps=0.001, momentum=0.1, affine=True)  
        self.relu = nn.ReLU(inplace=True)  

    def forward(self, x):  
        x = self.conv(x)  
        x = self.bn(x)  
        x = self.relu(x)  
        return x  

# Block35  
class Block35(nn.Module):  
    def __init__(self, scale=1.0):  
        super().__init__()  
        self.scale = scale  
        self.branch0 = FacenetLayer(256, 32, kernel_size=1)  
        self.branch1 = nn.Sequential(  
            FacenetLayer(256, 32, kernel_size=1),  
            FacenetLayer(32, 32, kernel_size=3, padding=1)  
        )  
        self.branch2 = nn.Sequential(  
            FacenetLayer(256, 64, kernel_size=1),  
            FacenetLayer(64, 64, kernel_size=3, padding=1),  
            FacenetLayer(64, 64, kernel_size=3, padding=1)  
        )  
        self.conv2d = nn.Conv2d(128, 256, kernel_size=1)  
        self.relu = nn.ReLU(inplace=False)  

    def forward(self, x):  
        y0 = self.branch0(x)  
        y1 = self.branch1(x)  
        y2 = self.branch2(x)  
        out = torch.cat((y0, y1, y2), 1)  
        out = self.conv2d(out)  
        out = out * self.scale + x  
        out = self.relu(out)  
        return out  

# Block17  
class Block17(nn.Module):  
    def __init__(self, scale=1.0):  
        super().__init__()  
        self.scale = scale  
        self.branch0 = FacenetLayer(896, 128, kernel_size=1)  
        self.branch1 = nn.Sequential(  
            FacenetLayer(896, 128, kernel_size=1),  
            FacenetLayer(128, 128, kernel_size=(1,7), padding=(0,3)),  
            FacenetLayer(128, 128, kernel_size=(7,1), padding=(3,0))  
        )  
        self.branch2 = nn.Sequential(  
            FacenetLayer(896, 128, kernel_size=1),  
            FacenetLayer(128, 128, kernel_size=3, padding=1),  
            FacenetLayer(128, 128, kernel_size=3, padding=1)  
        )  
        self.conv2d = nn.Conv2d(384, 896, kernel_size=1)  
        self.relu = nn.ReLU(inplace=False)  

    def forward(self, x):  
        y0 = self.branch0(x)  
        y1 = self.branch1(x)  
        y2 = self.branch2(x)  
        out = torch.cat((y0, y1, y2), 1)  
        out = self.conv2d(out)  
        out = out * self.scale + x  
        out = self.relu(out)  
        return out
    
# Block8
class Block8(nn.Module):  
    def __init__(self, scale=1.0, noReLU=False):  
        super().__init__()  
        self.scale = scale  
        self.noReLU = noReLU  
        self.branch0 = FacenetLayer(1536, 192, kernel_size=1)  
        self.branch1 = nn.Sequential(  
            FacenetLayer(1536, 192, kernel_size=1),  
            FacenetLayer(192, 192, kernel_size=(1,3), padding=(0,1)),  
            FacenetLayer(192, 192, kernel_size=(3,1), padding=(1,0))  
        )  
        self.conv2d = nn.Conv2d(384, 1536, kernel_size=1)  
        if not self.noReLU:  
            self.relu = nn.ReLU(inplace=False)  

    def forward(self, x):  
        y0 = self.branch0(x)  
        y1 = self.branch1(x)  
        out = torch.cat((y0, y1), 1)  
        out = self.conv2d(out)  
        out = out * self.scale + x  
        if not self.noReLU:  
            out = self.relu(out)  
        return out
    
class InceptionResNetV2(nn.Module):
    def __init__(self, num_classes=1000, embedding_size=128):
        super(InceptionResNetV2, self).__init__()
        # (Define all layers as above...)

        # CNN backbone
        self.conv2d_1a = FacenetLayer(3, 32, kernel_size=3, stride=2)
        self.conv2d_2a = FacenetLayer(32, 32, kernel_size=3)
        self.conv2d_2b = FacenetLayer(32, 64, kernel_size=3, padding=1)
        self.maxpool_3a = nn.MaxPool2d(3, stride=2)
        self.conv2d_3b = FacenetLayer(64, 80, kernel_size=1)
        self.conv2d_4a = FacenetLayer(80, 192, kernel_size=3)
        self.conv2d_4b = FacenetLayer(192, 256, kernel_size=3, stride=2)
        
        self.repeat_1 = nn.Sequential(
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17),
            Block35(scale=0.17)
        )

        # Mixed_6a
        self.mixed_6a = nn.Sequential(
            FacenetLayer(256, 384, kernel_size=3, stride=2),
            nn.Sequential(
                FacenetLayer(256, 192, kernel_size=1),
                FacenetLayer(192, 192, kernel_size=3, padding=1),
                FacenetLayer(192, 256, kernel_size=3, stride=2)
            ),
            nn.MaxPool2d(3, stride=2)
        )

        # 10 Block17
        self.repeat_2 = nn.Sequential(
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10),
            Block17(scale=0.10)
        )

        # Mixed_7a
        self.mixed_7a = nn.Sequential(
            FacenetLayer(896, 384, kernel_size=3, stride=2),
            nn.Sequential(
                FacenetLayer(896, 192, kernel_size=1),
                FacenetLayer(192, 192, kernel_size=3, padding=1),
                FacenetLayer(192, 256, kernel_size=3, stride=2)
            ),
            nn.MaxPool2d(3, stride=2)
        )

        # 5 Block8
        self.repeat_3 = nn.Sequential(
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20),
            Block8(scale=0.20)
        )

        # Final Block8 with no ReLU
        self.block8 = Block8(scale=0.20, noReLU=True)

        # Final layers
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(0.6)
        # This is the embedding layer for feature extraction
        self.last_linear = nn.Linear(1536, embedding_size)
        # Optional: Add a final classification layer
        # self.classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x):
        # CNN backbone layers
        x = self.conv2d_1a(x)
        x = self.conv2d_2a(x)
        x = self.conv2d_2b(x)
        x = self.maxpool_3a(x)
        x = self.conv2d_3b(x)
        x = self.conv2d_4a(x)
        x = self.conv2d_4b(x)
        # Repeat blocks
        x = self.repeat_1(x)
        x = self.mixed_6a(x)
        x = self.repeat_2(x)
        x = self.mixed_7a(x)
        x = self.repeat_3(x)
        x = self.block8(x)
        # Pool
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        # Dropout
        x = self.dropout(x)
        # Final embedding layer
        embeddings = self.last_linear(x)
        return embeddings