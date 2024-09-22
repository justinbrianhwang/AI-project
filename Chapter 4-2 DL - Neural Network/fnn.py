## fnn.py
# FNN (Fully-connected Neural Network, 완전 연걸 신경망)

# Visual C++ -> 이 라이브러리를 다운 받아야 함.
# https://aka.ms/vs/17/release/vc_redist.x64.exe

def p(str):
    print(str, '\n')

# 라이브러리 임포트
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as layers
from tensorflow.keras.datasets import mnist

# 데이터 셋 로딩 train/test 분리
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 모델 생성
network = Sequential()

# 모델에 레이어 추가
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28,)))
network.add(layers.Dense(10, activation='softmax'))

# 모델 컴파일
network.compile(
    loss='categorical_crossentropy',
    optimizer='rmsprop',
    metrics=['accuracy']
)

# 차원 변경
train_images = train_images.reshape((60000, 28 * 28))
test_images = test_images.reshape((10000, 28 * 28))

# 0~1 값으로 스케일링
train_images = train_images.astype('float32') / 255
test_images = test_images.astype('float32') / 255

# 픽셀의 유무로 범주 분리
from tensorflow.keras.utils import to_categorical
train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# 신경망 훈련
network.fit(train_images, train_labels, epochs=5, batch_size=128)










































































