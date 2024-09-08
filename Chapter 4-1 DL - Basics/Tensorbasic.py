## 텐서 플로우 기본
# tensorbasic.py

def p(str):
    print(str, '\n')

# 텐서플로우 라이브러리
import tensorflow as tf

# 텐서 플로우 버전 확인
# p(f'텐서플로우 버전 : {tf.__version__}')

# 정수형 상수 텐서 -> 값바꾸면 에러!
tensor_int = tf.constant(5, dtype=tf.int32) # 5의 값을 가진 4바이트 정수 상수 텐서
p(tensor_int)

tensor1 = tf.constant([5, 3]) # shape가 2라 되어 있다.
p(tensor1)

# 정수형 변수 텐서 -> 값을 바꿔도 문제가 없음!
tensor2 = tf.Variable(5)
p(tensor2)
tensor2.assign(10)
p(tensor2)

# 넘파이 배열을 텐서로 변환
import numpy as np
numpy_arr = np.array([1, 2, 3, 4]) # 넘파이 배열
tensor_numpy = tf.convert_to_tensor(numpy_arr)
p(tensor_numpy)
p(tensor_numpy.dtype) # int32 -> 텐서의 datatype이 나옴

# 리스트를 텐서로 변환
li = [1, 3, 3.14, 7, 10]
tenor_list = tf.convert_to_tensor(li)
p(tenor_list)

# 텐서를 넘파이 배열로 변환
numpy_ar1 = tensor_numpy.numpy()
numpy_ar2 = tenor_list.numpy()
p(numpy_ar1)
p(numpy_ar2)

# 텐서플로우 함수
# 두 정수형 텐서 생성
tensor1 = tf.constant(5, dtype=tf.int32)
tensor2 = tf.constant(7, dtype=tf.int32)

# 텐서들끼리 덧셈 연산
result = tf.add(tensor1, tensor2)
p(result) # 상수형 텐서를 가진 값을 더해줌

# 텐서 곱셈 연산
matrix1 = tf.constant([[1, 2], [3, 4]])
matrix2 = tf.constant([[5, 6], [7, 8]])
result = tf.matmul(matrix1, matrix2)
p(result) # 행렬 곱셈의 결과로 나온다.

# 텐서 슬라이싱 연산

# 2차원 텐서 생성
tensor = tf.constant([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
slice_tensor = tensor[0:2, 1:3]
p(slice_tensor)

# 넘파이로 변경해서 출력 -> 넘파일 배열이 완료되었다.
p(slice_tensor.numpy())

## 활성화 함수 (Activation function)
'''
    - 인공 신경망에서 각각의 뉴런의 출력을 결정하는 함수 -> 이를 활성화 함수라고 한다. 
    - 비선형 데이터를 추가해서 신경망이 복잡한 모델링을 하거나 
      다양한 문제를 해결할 수 있도록 하는데 사용되는 함수이다.
    - ReLU(Recitified Linear Unit)
      은닉층에서 주로 사용되고, 양수인 경우 양수를 출력
      음수의 경우, 0을 출력 
'''

# 실수 tensor 생성
tensor = tf.constant([-5.14, 2.51, -4.14, -0.05])
relu = tf.nn.relu(tensor) # relu
p(f'활성화 함수 적용 결과: {relu.numpy()}') # 양수는 그대로 출력을 해줬으며, 음수는 0으로 바뀌주었다.

# 텐서 크기 변경 : 1차원 텐서를 2차원으로 변경!!
tensor = tf.constant([1, 2, 3, 4, 5, 6])
tensor_re = tf.reshape(tensor, (2, 3))
p(tensor_re)

# 모든 요소가 0인 텐서
zeros_tf = tf.zeros((5, 2))
p(zeros_tf) # 5행 2열이 모두 0인 영행렬을 만들었다!

# 모든 요소가 1인 텐서
ones_tf = tf.ones((5, 2))
p(ones_tf) # 5행 2열이 모두 1인 행렬을 만들었다!

# 주어진 값으로 요소를 채운 텐서
fill_tf = tf.fill((3,2), 10)
p(fill_tf) # 3행 2열인데, 모든 값이 10으로 채워진 행렬을 만들었다!

# 정규 분포에서 난수 생성하는 함수
tensor = tf.random.normal((3, 4), mean = 0.0, stddev = 1.0)
p(tensor) # 3행 4열이며, 평균이 0이고 표준편차가 1인 행렬을 만들었다.
# 참고로, 텐서에서는 행렬이라는 말보다는 차원이라는 말을 쓴다.











































