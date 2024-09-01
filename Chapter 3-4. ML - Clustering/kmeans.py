## kmeans.py

def p(str):
    print(str, '\n')


# K-Means 클러스터링
# 라이브러리 로딩
import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings(action='ignore')

# random seed 설정
np.random.seed(42)

# 중심점이 4개인 150개의 점 데이터를 무작위로 생성
points, labels = make_blobs(
    n_samples=150,   # 생성 데이터 수
    centers=4,       # 중심점 수
    n_features=2,    # 변수의 수
    random_state=42  # 렌덤 시드값
)

# 무작위로 생성된 점의 좌표 10개 출력
# print(points.shape, '\n', points[:10])
# print(labels.shape, '\n', labels[:10])

# 데이터 프레임
points_df = pd.DataFrame(points, columns=['X', 'Y'])
p(points_df)

# 그래프 설정 및 그리기
figure = plt.figure(figsize=(10, 6))
axes = figure.add_subplot(111)
axes.scatter(points_df['X'], points_df['Y'], label='Random Data')
# axes.grid()
# axes.legend()
# plt.show()


# K-means 클러스터 생성

#kmeans 라이브러리
from sklearn.cluster import KMeans

#클러스터 생성
k_cluster = KMeans(n_clusters=4) # 4개의 클러스터 -> 사실 우리가 해줄 수 있는건 이거 하나이다.

# 클러스터에 데이터 입력해서 학습
k_cluster.fit(points)

# p(k_cluster.labels_) # 레이블
# p(np.shape(k_cluster.labels_)) # 150개
# p(np.unique(k_cluster.labels_)) # 레이블들의 unique 값

# 색상 딕셔너리
color_di = {
    0: 'red',
    1: 'blue',
    2: 'green',
    3: 'black'
}

# 그래프 그리기
# plt.figure(figsize=(10, 6))
# for cluster in range(4):
#     cluster_sub = points[k_cluster.labels_ == cluster]
#     plt.scatter(
#         cluster_sub[:, 0],                  # 첫 번째 feature
#         cluster_sub[:, 1],                  # 두 번째 feature
#         c = color_di[cluster],              # 클러스터별 색상
#         label = f"Cluster {cluster}"        # 클러스터별 레이블
#     )
# plt.grid(True)
# plt.legend()
# plt.show()


## K-Means 원형 클러스터 생성

# 라이브러리
from sklearn.cluster import KMeans
from sklearn.datasets import make_circles

# n_samples: 샘플 수
# factor : 원 안의 원과 원 바깥의 원의 크기 비율로
#          값이 작을 수록 안쪽 원이 작아짐, 값이 커질수록 안쪽 원이 커짐
#noise : 값이 0에 가까울수록 노이즈가 적고 값이 높을수록 노이즈가 많음
circle_point, circle_labels = make_circles(n_samples=150, factor=0.5, noise=.01)

# 그래프 크기
plt.figure(figsize=(10, 6))

# 모델 생성
circles_kmeans = KMeans(n_clusters=2)

# 모델 학습
circles_kmeans.fit(circle_point)

# 색성 딕셔너리
color_di = {0: 'blue', 1: 'red'}

# 스캐터
for i in range(2):
    cluster_sub = circle_point[circles_kmeans.labels_ == i]
    plt.scatter(
        cluster_sub[:, 0],
        cluster_sub[:, 1],
        c = color_di[i],
        label = f"cluster_{i}"
    )
plt.grid(True)
plt.legend()
plt.show()

# 클러스터별 색상 적용하여 스캐터 그리기
# for cluster in range(3):
#     cluster_sub = X[diag_kmeans.labels_ == cluster]
#     axes.scatter(cluster_sub[:, 0], cluster_sub[:, 1], c=color_di[cluster],
#                label=f"cluster {cluster}")
# plt.legend()
# plt.show()

X, y = make_blobs(n_samples=200, random_state=163)
# p(X)
# p(y)

# 변환 행렬
transformation = [[0.6, -0.6], [-0.3, 0.8]]

# dot() : 배열과 행렬사이의 행렬곱셈을 수행할 수 있고 X배열의 모든 데이터 포인트에 대한 선형변환을 적용시킬 수 있다
diag_points = np.dot(X, transformation)
#p(diag_points)

# 그래프 사이즈
#figure = plt.figure(figsize=(10, 6))

# 서브플랏 추가
#axes = figure.add_subplot(111)

# 클러스터 생성
#diag_kmeans = KMeans(n_clusters=3)

# 모델 학습
#diag_kmeans.fit(X)

# 색상 딕셔너리
#color_di = {0: "red", 1: "blue", 2: "green"}

# 클러스터별 색상 적용하여 스캐터 그리기
#for cluster in range(3):
#    cluster_sub = X[diag_kmeans.labels_ == cluster]
#    axes.scatter(cluster_sub[:, 0], cluster_sub[:, 1], c=color_di[cluster],
#                 label=f"cluster {cluster}")
#plt.legend()
#plt.show()



## DBSCAN : 밀도 기반 클러스터링 알고리즘

# 라이브러리 로딩
from sklearn.cluster import DBSCAN

figure = plt.figure(figsize=(10, 6))
axes = figure.add_subplot(111)
color_di = {1: 'red', 1: 'blue', 2: 'green', 3: 'black', 4: 'orange'}

epsilon = 0.5 # 반경
minPts = 3 # 최소 인접 포인트 수

# DBSCAN 생성
# eps : 클러스터의 반경
# min_samples : 클러스터를 구성하기 위한 최고 인접포인트의 개수
# metric: 거리 측정 방법을 지정하는 매개변수, 기본적으로 유클리드 거리
# algorithm: 클러스터링 알고리즘을 선택하는 매개변수, 기본값은 auto
diag_dbscan = DBSCAN(eps=epsilon, min_samples=minPts) # 생성

# 학습
diag_dbscan.fit(diag_points)

# 클러스터의 수, DBSCAN의 클러스터번호는 음수값을 포함하므로 +1 해줌
n_cluster = max(diag_dbscan.labels_) + 1
# p(n_cluster)

# 각 데이터 포인트의 클러스터 번호
# -1: noise, 0~4: 클러스터 번호
p(diag_dbscan.labels_)

# 스캐터
figure = plt.figure(figsize=(10, 6))
axes = figure.add_subplot(111)
color_di = {0:'red', 1:'blue', 2:'green', 3:'black', 4:'orange'}
for i in range(n_cluster):
    cluster_sub = diag_points[diag_dbscan.labels_ == i]
    plt.scatter(
        cluster_sub[:, 0],
        cluster_sub[:, 1],
        c = color_di[i]
    )
axes.grid(True)
plt.show()









