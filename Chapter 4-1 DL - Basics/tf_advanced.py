# tf_advanced.py
def p(str):
    print(str, '\n')

# 라이브러리
import tensorflow as tf
from tensorflow.keras.layers import Dense, Flatten, Conv2D
from tensorflow.keras import Model

# mnist 데이터 셋
mnist = tf.keras.datasets.mnist

# 트레인/테스트 분리
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 실수화
x_train, x_test = x_train / 255.0, x_test / 255.0

# 차원 추가
x_train = x_train[..., tf.newaxis].astype('float32')
x_test = x_test[..., tf.newaxis].astype('float32')

# 데이터셋 섞고 배치 생성
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)) \
    .shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)



# 모델클래스 생성
class MyModel(Model):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = Conv2D(32, 3, activation='relu')
        self.flatten = Flatten()
        self.d1 = Dense(128, activation='relu')
        self.d2 = Dense(10)

    def call(self, x):
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.d1(x)
        return self.d2(x)


# 모델 생성
model = MyModel()

# 활성화 함수, 손실 함수
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)

# 트레인/테스트 활성화, 손실
train_loss = tf.keras.metrics.Mean(name='train_loss')
train_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='train_accuracy')
test_loss = tf.keras.metrics.Mean(name='test_loss')
test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')


# 모델 훈련
@tf.function
def train_step(images, labels):
    with tf.GradientTape() as tape:
        predictions = model(images, training=True)
        loss = loss_object(labels, predictions)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    train_loss(loss)
    train_accuracy(labels, predictions)


# 모델 테스트
@tf.function
def test_step(images, labels):
    predictions = model(images, training=False)
    t_loss = loss_object(labels, predictions)
    test_loss(t_loss)
    test_accuracy(labels, predictions)


# 회수
EPOCHS = 5

for epoch in range(EPOCHS):
    p('epoch')
    train_loss.reset_state()
    train_accuracy.reset_state()
    test_loss.reset_state()
    test_accuracy.reset_state()
    for images, labels in train_ds:
        train_step(images, labels)
    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)
    print(
        f'Epoch {epoch + 1}, '
        f'Loss : {train_loss.result()}, '
        f'Accuracy : {train_accuracy.result() * 100}, '
        f'Test Loss : {test_loss.result()}, '
        f'Test Accuracy : {test_accuracy.result() * 100}'
    )




































