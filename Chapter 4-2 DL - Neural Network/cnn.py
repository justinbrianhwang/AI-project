## cnn.py
# cnn을 활용한 이미지 분류 작업

def p(str):
    print(str, '\n')

# 라이브러리 임포트
import tensorflow as tf
from tensorflow.keras import layers, datasets, models
import matplotlib.pyplot as plt

# MNIST 데이터 셋 로딩 : 10개의 클래스 (분류)로 되어 있는 6만개 컬러이미지
#                      5만개는 훈련용, 1만개는 테스트 용
(train_images, train_labels), (test_images, test_labels) = \
    datasets.cifar10.load_data()

# 데이터 정규화 : 픽셀 값들이 0에서 1사이에 위치하도록 정규화
train_images, test_images = train_images / 255.0, test_images / 255.0

# 클래스 명 10개
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(10, 10))
for i in range(25):
    plt.subplot(5, 5, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(train_images[i])
    plt.xlabel(class_names[train_labels[i][0]])
plt.show()

# 모델 생성, 합성 곱 레이어 생성
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

# 모델 요약 정보
model.summary()

# 모델에 Dense 층 추가
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10))

model.summary()

# 모델 컴파일
model.compile(
    optimizer='adam', # 활성화 함수
    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), # 손실함수
    metrics=['accuracy'] # 정확도
)

# 모델 훈련 -> fit 함수는 훈련하는 함수이다.
history = model.fit(
    train_images,
    train_labels,
    epochs=10, # 훈련 횟수
    validation_data=(test_images, test_labels) # 검증(테스트) 데이터
)

# 모델 평가
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0.5, 1]) # y축의 범례
plt.legend(loc='lower right') # 범례 위치
test_loss, test_acc = model.evaluate(test_images, test_labels, verbose=2)
plt.show()

# 정확도, 손실 출력
p(f'정화도: {test_acc}, 손실: {test_loss}')
