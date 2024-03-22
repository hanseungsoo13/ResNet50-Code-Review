![image](https://github.com/hanseungsoo13/ResNet50-Code-Review/assets/75753717/aea942c8-60d6-4056-8028-d73926010fc2)# ResNet50-Code-Review

안녕하세요! 그동안 논문을 리뷰만 노션에 정리해왔고 코드로는 따로 구현을 못하고 있었는데, 이제부터 찬찬히 Pytorch를 다시 익혀보면서 논문 리뷰에 코드 리뷰를 곁들여 해보려고 합니다.

제가 첫번째로 리뷰할 논문은 2015년 마이크로소프트에서 발표한 'Deep Residual Learning for Image Recognition' 입니다. 논문에서 skip connection을 활용한 ResNet을 제안하였으며, 오늘 직접 구현까지 해보면서 살펴보겠습니다.

## Paper Review
저자가 논문을 통해서 발표하면서 던진 질문은 다음과 같습니다.

Deep한 모델이 무조건적으로 좋은 모델일까?

물론 이 논문이 발표되기 전까지는 점점 Deep해질수록 성능이 좋은 CNN모델로써 의미가 있었습니다. 그러나 단순히 층만 많이 쌓은 Deep한 모델은 Gradient Vanishing 혹은 Gradient Exploding 문제에 빠지며 역설적이게도 accuracy가 낮아지고 degrade되는 경우가 많이 생겨났습니다.

![image](https://github.com/hanseungsoo13/ResNet50-Code-Review/assets/75753717/57d8a598-0b9d-4eb9-82cd-cd5dc348d1cb)
<그림1> 출처: Deep Residual Learning for Image Recognition Figure 1.
위의 그래프를 보면 56개의 layer를 쌓은 더 deep한 모델이 20 layer를 쌓은 덜 deep한 모델보다 training과 test error의 비율이 더 높음을 알 수가 있습니다.

이와 같은 문제점을 바탕으로 논문에서는 더 deep한 모델보다 덜 deep하면서 error을 줄일 수 있는 구조를 발명하기 위해 노력합니다.

### Deep Residual Learning
![image](https://github.com/hanseungsoo13/ResNet50-Code-Review/assets/75753717/7e3ec65e-d354-489c-8a6d-09ab9884b067)
<그림2> 출처: Deep Residual Learning for Image Recognition Figure 2.
Residual Learning은 위에서 저자가 던진 논문에서 정답을 던져준 핵심 아이디어입니다. 기존 plain model이 함수 F(x)와 같다고 할 때 함수의 input으로 들어갔던 x가 identity mapping을 통해서 다시 더해져 F(x)+x가 output되는 구조입니다.

저자는 이 구조를 shortcut connection이라고 불렀습니다. shortcut connection은 단순히 input이 몇 개의 layer를 건너뛰었다가 다시 합쳐지는 과정인데, 이 과정에서 어떠한 parameter 증가나 연산이 들어가지 않기 때문에 비용적으로 부담되거나 모델의 복잡성 문제와도 자유롭게 활용이 가능한 "치트키" 같은 느낌이었을 겁니다.

![image](https://github.com/hanseungsoo13/ResNet50-Code-Review/assets/75753717/85e7a272-a36c-4ed0-9977-d339d6f37bae)
<그림3> 출처: Deep Residual Learning for Image Recognition Figure 3.

위에서의 그림이 Plain Network와 Residual Learning을 활용한 Network의 차이를 보여준다. 사실 매우 단순합니다. 그냥 plain network에 shortcut connection이 반복적으로 추가되면 Residual Learning Network가 되는 것 입니다. 이 구조에 대해서는 코드 리뷰를 해보면서 다시 확인해보면 좋을 것 같습니다.

## Code Review
코드를 실제 데이터셋에 적용시킨 것은 Github에 적용해서 올려두겠습니다. 이곳에서는 ResNet구현에 핵심이 되는 Network에 대한 설명만 작성하도록 하겠습니다.

우선 논문에서 ResNet은 layer의 수에따라 버전이 5개가 소개가 됩니다. 18-layer부터 152-layer까지 있지만, 이번 리뷰에서는 가장 대표적인 ResNet-50을 구현해보겠습니다.

![image](https://github.com/hanseungsoo13/ResNet50-Code-Review/assets/75753717/a3bd09bd-16af-4103-8420-80857fa6f158)
<그림4> 출처: Deep Residual Learning for Image Recognition Table 1.

우선 코딩을 하기 전에 어떤 식으로 구현해야할지 구상을 먼저 해보겠습니다.
위의 표에서 각 칸이 하나의 Block을 형성하고 있는 것 같습니다. 각 Block의 구조나 반복횟수는 ResNet의 종류마다 다르지만, 그래도 이 Block을 잘 활용하면 쉽게 구현할 수 있어 보입니다. 이 Block도 자세히보면 크게 2가지 종류로 나눌 수 있겠습니다. 상대적으로 layer가 적은 Block은 3*3conv layer만 활용하고, 50층 이상의 layer를 가지는 ResNet은 1*1,3*3,1*1 conv layer순으로 Block이 구성되어 있습니다.

![image](https://github.com/hanseungsoo13/ResNet50-Code-Review/assets/75753717/95eb71e7-d4a8-48e4-99ae-be36255c0165)
<그림5> 출처: Deep Residual Learning for Image Recognition Figure 5.

위에서 살펴본 대로 우선 50-layer의 Block을 만들고, 이 Block의 반복과 shortcut connection을 이용해 ResNet-50을 구현 해보겠습니다.

### 1. BottleNeckBlock
1*1 conv 연산이 있기에 BottleNeck Block으로 이름지었습니다.

class BottleNeck(nn.Module):
    expansion=4
    def __init__(self,in_channels,out_channels,stride=1):
        super().__init__()
        
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels,kernel_size=3,stride=stride,padding=1,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels,out_channels*BottleNeck.expansion,kernel_size=1,stride=1,bias=False),
            nn.BatchNorm2d(out_channels*BottleNeck.expansion),
        )

        if stride != 1 or in_channels != BottleNeck.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion))
        else:
            self.shortcut = nn.Sequential()
            
        self.relu = nn.ReLU()
        

    def forward(self,x):
        x = self.residual_function(x) + self.shortcut(x)
        x = self.relu(x)
        return x
위의 코드에서 residual function이 Block의 convolution network, shortcut이 shortcut connection을 담당하는 부분입니다.
Block의 구조와 동일하게 convolution layer를 쌓아주고, Block의 마지막에 shortcut을 더해주면서 Residual Learning이 가능한 구조로 만들어 주었습니다.

### 2. ResNet
앞서 만들었던 Block module을 이용해서 최종적인 ResNet module을 생성해줍니다.

class ResNet(nn.Module):
    def __init__(self,Block,num_block,num_classes=10):
        super().__init__()
        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3,64,kernel_size=7,stride=2,padding=3,bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3,stride=2,padding=1),
        )
        self.conv2 = self.make_layer(Block,num_block[0],64,stride=1)
        self.conv3 = self.make_layer(Block,num_block[1],128,stride=2)
        self.conv4 = self.make_layer(Block,num_block[2],256,stride=2)
        self.conv5 = self.make_layer(Block,num_block[3],512,stride=2)

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
        self.fc = nn.Linear(512*Block.expansion,num_classes)
        
    #Block이 반복되면서 input,output size에 변동이 생겨 이를 연결해주는 과정
    def make_layer(self,Block,num_block,out_channels,stride):
        layers=[]
        for i in range(num_block):
            if i == 0:
                st=stride #block이 시작할 때마다 stride를 변경하여 input size 조정
                
            else: 
                st=1
            layers.append(Block(self.in_channels,out_channels,stride=st))
            self.in_channels=out_channels*Block.expansion
        
        return nn.Sequential(*layers)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0),-1)
        x = self.fc(x)
        return x
    
논문에서 Block을 통과할 때마다 image size는 절반이되고, filter의 수는 4배씩 늘리는 구조를 제안했기 때문에 이 부분을 맞춰주는데 조금 신경을 써줘야 했습니다.
그래서 stride를 늘려야 하는 타이밍, input size와 output size가 잘 연결될 수 있도록 make layer 함수에 반복문과 조건문을 섞어서 쎃어줬습니다.

def resnet50():
    return ResNet(BottleNeck, [3,4,6,3])
가장 마지막으로 ResNet50에 맞는 Block과 반복횟수를 넣어주면 구현 완료입니다.
아래는 ResNet50의 마지막 블럭의 layer 구조이며 VGG16과 같은 이미지모델과 비교했을 때 상대적으로 가벼운 모델임을 알 수 있습니다.

![image](https://github.com/hanseungsoo13/ResNet50-Code-Review/assets/75753717/87c6de31-2ce5-4451-89da-b1d718352e69)

이상으로 ResNet에 대한 논문 리뷰와 코드 리뷰를 마쳤습니다. 이제 가장 기본이 되는 모델을 구현했기 때문에 아직 갈길이 멀다고 생각됩니다. 꾸준히 하면서 최신 논문까지 구현할 수 있는 수준으로 코딩 실력을 올려보겠습니다. 긴 글 읽어주셔서 감사합니다.

참고한 자료입니다.

Paper: Deep Residual Learning for Image Recognition
- https://paperswithcode.com/paper/deep-residual-learning-for-image-recognition
Code
- github.com/weiaicunzai/pytorch-cifar100/blob/master/models/resnet.py
