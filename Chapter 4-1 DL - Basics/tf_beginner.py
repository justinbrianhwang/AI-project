# tf_beginner.py

def p(str):
    print(str, '\n')

# 라이브러리
import tensorflow as tf

# mnist 데이터 셋 로딩
mnist = tf.keras.datasets.mnist

# 트레이닝셋/테스트셋 분리
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# x_train과 X_test를 실수화!
x_train, x_test = x_train / 255.0, x_test / 255.0

# Squaltial 모델
# 케라스의 레이어들을 순차적으로 적용하는 모델
model = tf.keras.models.Sequential(
    [
        tf.keras.layers.Flatten(input_shape=(28, 28)),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='softmax')
    ]
)

# 모델 컴파일
# 활성화 함수, 손실함수, 결과를 지정
model.compile(
    optimizer='adam', # 활성화 함수
    loss='sparse_categorical_crossentropy', # 손실 함수
    metrics=['accuracy'] # 결과는 정확도
)

# 예측값
predictions = model(x_train[:1]).numpy()
p(predictions)

# 예측값들을 확률로 변환
p(tf.nn.softmax(predictions).numpy())

# 손실함수 정의
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)
p(loss_fn(y_train[:1], predictions).numpy()) # 계속 바뀐다.

# 모델 컴파일
model.compile(
    optimizer='adam',
    loss=loss_fn,
    metrics=['accuracy']
)

# 모델 훈련
# epochs : 횟수 -> 이 수를 늘리면 훈련을 어려번 시키는 것이다. => 이것은 CPU가 처리에겐 무리이다. GPU 시켜야 한다.
model.fit(x_train, y_train, epochs=5)

# 예측 모델
probablity_model = tf.keras.Sequential([
    model,
    tf.keras.layers.Softmax()
])

# 예측값
p(probablity_model(x_test[:5]))














































