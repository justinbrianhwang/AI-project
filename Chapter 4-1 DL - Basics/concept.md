# In English


## DL

Deep Learning: A field of machine learning, but on a higher level. While machine learning (ML) involves providing data, creating a model, and running an algorithm to get results, deep learning (DL) allows the machine to learn on its own, creating multiple models and figuring out the best one through self-learning.

### Deep Learning

- A subset of machine learning.
- Unlike machine learning, deep learning automatically decides and learns from various models (algorithms) to predict more accurate results.
- Example of deep learning:
    - AlphaGo: Learned from existing Go games and then derived its algorithms through self-learning.

### Deep Learning Techniques

1. Ensemble Techniques
    - Uses multiple base models to create one comprehensive model.
    - Improves prediction performance.
    1. Voting
        - Combines classifiers with different algorithms.
        - Hard Voting: Majority voting.
        - Soft Voting: Averages the probabilities of classifiers' predictions to choose the class with the highest probability.
    2. Bagging (Bootstrap Aggregating)
        - Reconstructs data to create diverse models.
        - Used in models prone to overfitting, like decision trees.
        - Random forest is a model based on bagging.
    3. Boosting
        - Assigns higher weights to data that previous models misclassified.
        - Adaptive Boosting: Focuses on misclassified data in each iteration.
        - Gradient Boosting: Uses gradient descent to minimize errors across multiple models.

### Deep Learning

1. Neural Networks
    - A machine learning algorithm that mimics the human brain.
    - Comprised of input, hidden, and output layers.
    - Deep neural networks (DNNs) consist of multiple layers.
    - AI > Machine Learning > Deep Learning.

1. What Neural Networks Can Do
    - Classification
    - Regression
    - Clustering
    - Image generation
    - Natural language processing (NLP)

1. Key Terms
    - GPU (Graphics Processing Unit): A high-performance parallel processing unit.
    - BERT (Bidirectional Encoder Representations from Transformers): A model developed by Google for NLP.
    - Encoder: Converts input sentences into meaningful vector representations for tasks like translation.
    - Perceptron: A fundamental unit in a neural network that outputs based on weighted inputs.

2. Key Points in Deep Learning
    - Data: Lots of high-quality data.
    - Model: CNN, RNN, FNN, Transformer, etc.
    - Algorithms: Gradient descent-based algorithms.
    - Loss Function: Measures model performance.

1. Deep Learning Models
    - CNN (Convolutional Neural Network): Used for image processing.
    - RNN (Recurrent Neural Network): Used for sequential data processing.
    - GAN (Generative Adversarial Network): A generative model that pits a generator against a discriminator.

### TensorFlow

An open-source machine learning framework developed by Google.

1. Tensor
    - The basic data type in TensorFlow.
    - Short for a multi-dimensional array.

1. Variable Tensor
    - A mutable tensor used for values that need to change during training, like weights.
    - Created using `tf.Variable()`.

1. Constant Tensor
    - Immutable values that cannot be changed after creation.
    - Created using `tf.constant()`.

1. Converting between Tensors and Numpy Arrays
    - Use `tf.convert_to_tensor()` to convert a numpy array into a tensor.
    - Use `.numpy()` to convert a tensor into a numpy array.

1. TensorFlow Functions
    - `tf.constant()`: Creates a constant tensor.
    - `tf.Variable()`: Creates a variable tensor.
    - `tf.add()`: Adds two tensors.
    - `tf.matmul()`: Performs matrix multiplication.
    - `tf.zeros()`: Creates a tensor filled with zeros.
    - `tf.ones()`: Creates a tensor filled with ones.
    - `tf.fill()`: Fills a tensor with a specific value.

1. Tensor Attributes
    - Dimensions, size, data type, and device (CPU/GPU) information.






# In korean

## DL

딥러닝: 머신러닝의 한 분야로, 한 단계 더 높은 개념이다. 머신러닝(ML)은 데이터를 제공하고 모델을 만들고 알고리즘을 짜면 결과가 나오지만, 딥러닝(DL)은 기계가 스스로 학습하고 여러 모델을 만들면서 최적의 모델을 찾는 과정을 포함한다.

### 딥러닝

- 머신러닝의 일종.
- 머신러닝과 달리 다양한 모델(알고리즘)을 스스로 학습하여 더 정확한 결과를 예측하는 학문/기술 분야.
- 딥러닝의 예:
    - 알파고: 기존의 바둑 기보 데이터를 학습한 후, 스스로 알고리즘을 도출함.

### 딥러닝 기법

1. 앙상블 기법
    - 여러 개의 기본 모델을 결합해 하나의 새로운 모델을 만드는 기법.
    - 예측 성능을 향상시키기 위해 사용.
    1. 보팅(Voting)
        - 서로 다른 알고리즘을 가진 분류기를 결합하는 방식.
        - 하드 보팅: 다수결의 원칙에 기반하여 다수의 분류기가 결정한 값을 최종 예측값으로 선택.
        - 소프트 보팅: 각 분류기의 예측 확률을 평균 내어 가장 높은 확률을 가진 클래스를 선택.
    2. 배깅(Bagging: Bootstrap Aggregating)
        - 데이터를 재구성하여 여러 모델을 만드는 앙상블 기법.
        - 의사결정나무처럼 과적합이 발생하기 쉬운 모델에 사용.
        - 랜덤 포레스트는 배깅을 기반으로 한 모델이다.
    3. 부스팅(Boosting)
        - 이전 모델이 잘못 예측한 데이터에 가중치를 더해 학습하는 방법.
        - Adaptive Boosting: 오분류된 데이터에 가중치를 부여하여 부스팅 수행.
        - Gradient Boosting: 경사 하강법을 통해 오류를 최소화하여 모델을 개선.

### 딥러닝

1. 신경망 (Neural Network)
    - 인간의 뇌 기능을 모방한 머신러닝 알고리즘.
    - 입력층, 은닉층, 출력층으로 구분되며, 층이 많아질수록 깊어진다.
    - 깊은 신경망을 **딥 뉴럴 네트워크(Deep Neural Network)**라고 한다.
    - 딥러닝은 이러한 신경망을 학습시키는 과정이다.

1. 신경망으로 할 수 있는 것들
    - 분류, 회귀, 클러스터링, 이미지 생성, 자연어 처리 등.

1. 관련 용어
    - GPU: **그래픽 처리 장치**, 고성능 병렬 처리 장치.
    - BERT: 구글에서 개발한 자연어 처리 모델로, 양방향 방식을 처음 도입한 모델.
    - 인코더: 입력 문장을 벡터로 변환하여 디코더에 전달하는 역할.
    - 퍼셉트론: 인공 신경망의 기본 단위로, 다수의 입력을 받아 하나의 출력을 생성.

2. 딥러닝 학습 포인트
    - 데이터: 양질의 **많은** 데이터.
    - 모델: CNN, RNN, FNN, Transformer 등.
    - 알고리즘: 경사 하강법(Gradient Descent)을 기반으로 한 다양한 알고리즘.
    - 손실 함수(Loss Function): 모델 성능을 측정하는 척도.

1. 딥러닝 모델
    - CNN(Convolutional Neural Network): 이미지 처리에 주로 사용되는 딥러닝 모델.
    - RNN(Recurrent Neural Network): 순차 데이터를 처리하는 딥러닝 모델.
    - GAN(Generative Adversarial Network): 생성자와 판별자가 경쟁하는 구조의 생성 모델.

### 텐서플로우(TensorFlow)

구글에서 개발한 오픈소스 머신러닝 프레임워크.

1. 텐서 (Tensor)
    - 다차원 배열을 의미하며, 텐서플로우에서 사용하는 기본 데이터 타입.

1. 변수 텐서 (Variable Tensor)
    - 훈련 중에 값이 변하는 가중치 등을 나타내기 위해 사용되는 텐서.
    - `tf.Variable()`로 생성한다.

1. 상수 텐서 (Constant Tensor)
    - 변경할 수 없는 값을 가지며, 모델의 하이퍼파라미터나 고정된 값을 나타낸다.
    - `tf.constant()`로 생성.

1. 넘파이 배열을 텐서로 변환
    - `tf.convert_to_tensor()`를 사용.

1. 텐서를 넘파이 배열로 변환
    - `.numpy()`로 변환 가능.

1. 텐서플로우 함수
    - `tf.constant()`: 상수 텐서 생성.
    - `tf.Variable()`: 변수 텐서 생성.
    - `tf.add()`: 두 텐서를 더하는 함수.
    - `tf.matmul()`: 두 텐서의 행렬 곱.
    - `tf.zeros()`: 모든 요소가 0인 텐서 생성.
    - `tf.ones()`: 모든 요소가 1인 텐서 생성.
    - `tf.fill()`: 주어진 값으로 텐서를 채우는 함수.

1. 텐서의 속성
    - 차원, 크기, 데이터 타입, 디바이스(CPU, GPU) 등.


