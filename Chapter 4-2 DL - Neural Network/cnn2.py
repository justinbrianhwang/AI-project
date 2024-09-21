## cnn2.py
# CNN 꽃 이미지 분류

def p(str):
    print(str, '\n')

# 라이브러리  로딩
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

# 데이터 로딩 : 3670장의 꽃 사진 데이터
# 3700장의 꽃 사진은 daisy, dandelion, roses, sunflowers, tulips 5개의 분류
import pathlib
dataset_url = 'https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz'
# flower_photos 폴더에 압축파일 해제
data_dir = tf.keras.utils.get_file('flower_photos', origin=dataset_url, untar=True)
data_dir = pathlib.Path(data_dir)
image_cnt = len(list(data_dir.glob('*/*.jpg'))) # 사진의 수
p(image_cnt)

# 장미 이미지 glob함수는 찾을 때 쓰는 것이다.
roses = list(data_dir.glob('roses/*'))
image = PIL.Image.open(str(roses[0]))
image.show()

# 튤립이미지
tulips = list(data_dir.glob('tulips/*'))
image2 = PIL.Image.open(str(tulips[0]))
image2.show()

# 데이터 셋 만들기
train_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, # 데이터 디렉터리
    validation_split=0.2, # 검증셋 20%
    subset='training',
    seed = 123,  # 랜덤 시드 값
    image_size = (180, 180), # 이미지 사이즈
    batch_size = 32 #
)

# 검증 set
val_ds = tf.keras.utils.image_dataset_from_directory(
    data_dir, # 데이터 디렉터리
    validation_split=0.2, # 검증셋 20%
    subset="validation",
    seed = 123,  # 랜덤 시드 값
    image_size = (180, 180), # 이미지 사이즈
    batch_size = 32 #
)

# 클래스명
class_names = train_ds.class_names
p(class_names)

# 시각화
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
for images, labels in train_ds.take(1):
    for i in range(9):
        ax = plt.subplot(3, 3, i+1)
        plt.imshow(images[i].numpy().astype("uint8"))
        plt.title(class_names[labels[i]])
        plt.axis('off')
plt.show()

# 데이터 정규화
normalization_layer = layers.Rescaling(1./255) # 0~1 범위로
# 훈련데이터 각각에 정규화 실시
normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))

# 모델 생성
model = Sequential([
    layers.Rescaling(1./255, input_shape=(180, 180, 3)),
    layers.Conv2D(16, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(32, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Conv2D(64, 3, padding='same', activation='relu'),
    layers.MaxPooling2D(),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(5)
])

# 모델 요약 정보
model.summary()

# 모델 컴파일
model.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# 모델 훈련 시키기
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# 훈련 결과 시각화
acc = history.history['accuracy'] # 정확도
val_acc = history.history['val_accuracy'] # 검증 정확도
loss = history.history['loss'] # 손실
val_loss = history.history['val_loss'] # 검증 손실

plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(range(10), acc, label='Training Accuracy')
plt.plot(range(10), val_acc, label='validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(range(10), loss, label='Training Loss')
plt.plot(range(10), val_loss, label='validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


