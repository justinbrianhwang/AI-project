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

Deep Learning: 머신러닝의 한 분야이며, 한 차원 더 높은 것이라 생각하면 된다. 스스로 학습하는 것. ML은, 데이터를 주고 모델을 만들고 알고리즘을 짜면 결과가 나오지만, DL은 스스로 모델을 여러개 만들고 학습하는 것을 찾도록 만드는 것이다.

### Deep Learning

- 머신 러닝의 일종
- 머신 러닝과 달리 다양한 모델(알고리즘)에 대한 학습을 스스로 결정하고 학습해 나가서 좀 더 정확한 결과를 예측하는 학문/기술 분야
- 딥러닝의 예
    - 알파고 (기존의 바둑 기보 데이터들을 모두 학습하고 학습 결과를 통해서 스스로 알고리즘들을 도출해 냄)

### Deep Learning 기법

1. 앙상블(Ensemble) 기법
    - 여러 개의 기본 모델을 활용해서 하나의 새로운 모델을 생성해 내는 기법
    - 모델의 예측 성능을 향상시키기 위해서 사용
    1. 보팅(Voting)
        - 서로 다른 알고리즘을 가진 분류기를 결합하는 방식
        - 하드 보팅(Hard voting): 다수결의 원칙 기반, 예측의 결과값을 다수 분류기가 결정한 예측값을 최종 보팅 결과값으로 선택함
        - 소프트 보팅(Soft Voting): 레이블 값 결정 확률을 모두 구하고 평균을 내서 확률이 가장 높은 레이블 값을 최종 결과 값으로 선택함 
    2. 배깅(Bagging: Bootstrap Aggregating)
        - 데이터를 재구성하여 다양한 모델을 만들어내는 앙상블 기법 중 하나
        - 의사 결정 트리처럼 과 적합이 발생하기 쉬운 모델에서 사용
        - 랜덤 포레스트는 배깅을 기반으로 한 모델로 여러 개의 의사 결정 트리를 생성하고 그 결과를 조합하여 다양한 예측을 수행
    3. 부스팅(Boosting)
        - 이전 모델들이 잘못 예측한 데이터에 높은 가중치를 부여해서 학습하는 방법
        - Adaptive Boosting: 오분류된 데이터에 가중치를 부여하면서 부스팅을 수행
        - Gradient Boosting: 경사 하강법을 통해 오류를 최소화 하는 방법으로 모델 향상

### 딥러닝

1. 신경망 (Neural Network)
    - 머신러닝 알고리즘 중 하나
    - 인간의 뇌 기능을 흉내내서 만들어짐
    - 입력층, 은닉층, 출력층으로 구분
    - 층을 점점 늘려서 깊게 만든 **신경망을 심층신경망(Deep Neural Network)**이라 함
    - AI > 머신러닝 > 딥러닝

1. 신경망으로 할 수 있는 것들
    - 분류
    - 회귀
    - 클러스터링
    - 이미지 생성
    - 자연어처리

1. 관련 용어
    - GPU(Graphics Processing Unit): **고성능 병렬처리**를 위한 범용 연산장치
    - BERT(Bidirectional Encoder Representations from Transformers): 구글에서 개발한 자연어 처리 모델
    - 인코더 (Encoder): 입력 문장을 더 의미있는 벡터 표현으로 변환하여 디코더로 전달
    - 퍼셉트론 (Perceptron): 인공 신경망 모형의 하나로 다수의 신호를 받아서 하나의 신호를 출력

2. 딥러닝 학습 포인트
    - 데이터 : 양질의 **많은** 데이터
    - 모델 : CNN, RNN, FNN, Transformer …
    - 알고리즘 : Gradient Descent (경사 하강법)을 기초로 한 많은 알고리즘들…
    - 손실 함수(Loss Function) : 모델의 성능에 대한 척도

1. 딥러닝 모델
    - CNN(Convolutional Neural Network): 주로 이미지 처리에 사용되는 딥러닝 모델
    - RNN (Recurrent Neural Network): 순차적인 데이터 처리에 사용되는 딥러닝 모델
    - GAN(Generative Adversarial Network): 생성자와 판별자가 대립하며 경쟁하는 구조로 이미지를 생성, 감별

### 텐서플로우(TensorFlow)

구글에서 개발한 오픈소스 머신러닝 프레임워크

1. 텐서 (Tensor)
    - Tensor Multi-dimentional array의 줄임말 (다차원 배열)
    - 텐서플로우에서 사용하는 기본 데이터타입

1. 변수 텐서 (Variable Tensor)
    - 가변성 텐서로 생성 후 내부 값 수정이 가능함
    - tf.Variable() 클래스를 사용하여 생성

1. 상수 텐서
    - 불변성
    - 한 번 생성되면 내부 값을 변경할 수 없음

1. 넘파이 배열을 텐서로 변환
    - 변수  = tf.convert_to_tensor(넘파이 배열)

1. 텐서의 속성
    - 차원, 크기, 데이터 타입, 디바이스, 텐서 연산 등


