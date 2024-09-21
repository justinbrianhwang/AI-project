# In English

### Artificial neural networks

An artificial neural network modeled after the neural network system involved in the data input and output of neurons in the human brain. 

1. CNN (Convolutional Neural Network)
    - Deep learning models used primarily for image processing
    - Recognize local patterns and spatial structure using Convolution Layers
    - Applies multiple filters to the input data to extract the characteristics of the input image, and is designed to help you better understand the image.
    - Consists of Convolution, Activation, and Ploogng layers
    - Applications: face recognition, document analysis, climate pattern understanding, image classification, convolutional neural networks
    - The process of training a dataset using a CNN
        1. Importing datasets 
        2. Separate training data from test data → also used in ML so far
        3. Model organization: including convolution layers, pooling layers, connection layers, etc.
        4. Train a model 
        5. Evaluating models
    - Model configuration
        
         Create a Sequential Model
        
        Add a Conv2D layer: Layer that performs the convolution operation
        
        ex:
        
        Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding=’same’, activation=’relu’, input_shape=input_shape) 
        
        32: Number of output filters (how many feature maps to generate from that layer)
        
        kernel_size=(5, 5): size of the convolution curl 
        
        strides=(1, 1): interval to move over the kernel's image when performing convolution operations
        
        → (1, 1) to perform the operation, moving one pixel at a time
        
        padding='same': set padding method, if set to same, add padding to keep the size of the output feature map the same as the input image 
        
        activation='relu' : Use an activation function called relu → 0 if a negative number is entered, or use the entered value if a positive number is entered
        
        input_shape=: Input image shape (width, height, number of channels)
        
        A channel count of 1 results in a black and white image
        
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        Add a MaxPooling2D layer to your model
        
        pool_size=(2, 2) : specify the size of the pooling window 
        
        → Sliding over the input attribute map to perform the maximum pooling operation 
        
        → Divide the feature map into 2 $\times$ 2 regions using a pooling window of size (2, 2), downsampling by selecting the largest value in each region 
        
        strides=(2, 2): the interval the pooling window moves over the input image during the pooling operation 
        
        model.add(Flatten())
        
        Adding a Flatten layer to your model to convert multidimensional data to a one-dimensional vector 
        
        ex) (4, 4, 64) ⇒ convert to 1024 
        
        model.add(Dense(1000, activation=’relu’))
        
        Add a Dense layer to your model 
        
        1000: Number of neurons
        
        activation='relu': select activation function as relu
        

1. RNN (Recurrent Neural Network)
    - Deep learning models primarily used for sequential data processing and text
    - Has a cyclical structure, using outputs from previous time steps as inputs to the current time step
    - Applications: speech recognition, time series prediction, music composition, machine translation...








# In korean

### 인공 신경망

사람의 뇌에 있는 뉴런의 데이터 입출력에 관련된 신경망 체계를 본따 만든 인공적인 신경망 

1. CNN (Convolutional Neural Network)
    - 주로 이미지 처리에 많이 사용되는 딥러닝 모델
    - 컨볼루전 레이어 (Convolution Layer)를 사용해서 지역적인 패턴 및 **공간적인 구조**를 인식
    - 입력된 이미지의 특성들을 추출하기 위해 입력 데이터에 여러 필터를 적용하고, 이미지를 잘 이해할 수 있도록 설계되어 있음
    - 합성곱(Convolution), 활성화 함수(Activation), 풀링(Ploogng) 계층으로 이루어져 있음
    - 응용분야 : 얼굴 인식, 문서 분석, 기후 패턴 이해, 영상 분류, 컨볼루전 신경망
    - CNN을 사용해서 데이터셋을 학습하는 과정
        1. 데이터셋 불러오기 
        2. 학습 데이터와 테스트 데이터를 분리 → 여기까진 ML에서도 많이 사용 
        3. 모델 구성 : 컨볼루전 레이어, 풀링 레이어, 연결 레이어 등을 포함
        4. 모델 학습 시키기 
        5. 모델 평가
    - 모델 구성
        
         Sequential 모델 생성 
        
        Conv2D 레이어 추가 : convolution 연산을 수행하는 레이어
        
        코드 예)
        
        Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding=’same’, activation=’relu’, input_shape=input_shape) 
        
        32 : 출력 필터의 수 (해당 레이어에서 생성할 특성맵의 개수)
        
        kernel_size=(5, 5) : 컨볼루전 컬의 크기 
        
        strides=(1, 1) : 컨볼루전 연산을 수행할 때 커널의 이미지 위로 이동하는 간격 
        
        → (1, 1)로 값을 주면 한 픽셀씩 이동하며 연산을 수행 
        
        padding=’same’ : 패딩 방법을 설정, same으로 설정하면 출력 특성 맵의 크기를 입력한 이미지와 동일하게 유지하도록 패딩을 추가 
        
        activation=’relu’ : relu라는 활성화 함수를 사용 → 음수가 입력되면 0, 양수가 입력되면 입력된 값을 사용 
        
        input_shape= : 입력 이미지 모양 (넓이, 높이, 채널 수)
        
        채널수가 1이면 흑백 이미지
        
        model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        
        model에 MaxPooling2D 레이어를 추가
        
        pool_size=(2, 2) : 풀링 윈도우의 크기를 지정 
        
         →입력 특성맵 위를 슬라이딩하여 최대 풀링 연산을 수행함 
        
        → (2, 2) 크기의 풀링 윈도우를 사용하여 특성 맵을 2 $\times$ 2 영역으로 나누고, 각 영역에서 가장 큰 값을 선택해서 다운 샘플링을 함 
        
        strides=(2, 2) : 풀링 연산시에 풀링 윈도우가 입력 이미지 위를 이동하는 간격 
        
        model.add(Flatten())
        
        model에 Flatten 레이어를 추가하여 다차원 데이터를 1차원 벡터로 변환 
        
        ex) (4, 4, 64) ⇒ 1024로 변환 
        
        model.add(Dense(1000, activation=’relu’))
        
        model에 Dense 레이어를 추가 
        
        1000: 뉴런의 수 
        
        activation=’relu’ : 활성화 함수를 relu로 선택 
        

1. RNN (Recurrent Neural Network)
    - 순차적인 데이터 처리 및 텍스트에 주로 사용되는 딥러닝 모델
    - 순환적인 구조를 가지며, 이전 시간 단계의 출력을 현재 시간 단계의 입력으로 사용
    - 응용 분야 : 음성 인식, 시계열 예측, 음악 작곡, 기계 번역…






