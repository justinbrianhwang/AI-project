# tf_mnist.py
import keras
# 라이브러리 임포트
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential, load_model
from keras.datasets import mnist
from tensorflow.keras.layers import Input, Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.utils import to_categorical

# mnist 데이터 셋

img_rows = 28 # 행의 수
img_cols = 28 # 열의 수

# train/test 분리
(X_train, y_train), (X_test, y_test) = mnist.load_data()
print(X_train.shape, X_test.shape)

# 입력 크기
input_shape = (img_rows, img_cols, 1)
X_train = X_train.reshape(X_train.shape[0], img_rows, img_cols, 1)
X_test = X_test.reshape(X_test.shape[0], img_rows, img_cols, 1)
print(X_train.shape, X_test.shape)

# 실수 변환
X_train = X_train.astype('float32') / 255.0
X_test = X_test.astype('float32') / 255.0
print(X_train.shape[0], X_test.shape[0])

# 분류화
num_classes = 10 # 분류 10개
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)
print(y_train)
print(y_test)

# 모델 생성
model = Sequential()
model.add(Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same'\
                 , activation='relu', input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Conv2D(64, (2,2), activation='relu', padding='same'))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(1000, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# 모델 컴파일
model.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

# 모델 훈련
model.fit(X_train, y_train, batch_size=128, epochs=3\
          ,verbose=1 ,validation_data=(X_test, y_test))

# 손실 점수, 정확도
score = model.evaluate(X_test, y_test)
print(f'Test loss: {score[0]}')
print(f'Test accuracy: {score[1]}')

# 예측 값
predicted_result = model.predict(X_test)
predicted_label = np.argmax(predicted_result, axis=1)
test_label = np.argmax(y_test, axis=1)
print(test_label)

count = 0
plt.figure(figsize=(12, 8))
for i in range(16):
    count += 1
    plt.subplot(4, 4, count)
    plt.imshow(X_test[i].reshape(img_rows, img_cols), cmap='Greys', interpolation='nearest')
    tmp = 'Label: ' + str(test_label[i]), ', Prediction: ' + str(predicted_label[i])
    plt.title(tmp)
plt.tight_layout()
plt.show()


















































































