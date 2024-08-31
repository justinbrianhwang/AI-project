# English

## Clustering

Cluster: **A group of similar characteristics**

Clustering: **Grouping together elements** with similar characteristics.

### Purpose of Clustering

- **Faster computation** (If we analyze and classify every data feature each time, the **computation cost** increases.)
- Incorporate new data with similar characteristics into the cluster.
- Merge clusters to create larger clusters.

### Clustering Algorithms

1. $k$-Means Clustering 
    - An algorithm that groups data into $k$ clusters.
    - Each cluster has a center, and data points are assigned to the closest center.
    
    1. Process
        1. Randomly select cluster centers ($k$ refers to the number of clusters).
        2. Assign each data point to the closest center.
        3. Calculate the mean of the data points within each cluster and update the cluster centers.
        4. Repeat steps b and c until the center no longer changes.
        
        **The center no longer changes** = **Centroid** is fixed = All points have converged = **Cluster** is complete.
        
    2. Advantages
        - Simple and efficient algorithm.
        - Applicable to large datasets.
        - Easy to specify the number of clusters.
    3. Disadvantages
        - Assumes that clusters are **spherical**, making it difficult to cluster shapes that differ from a sphere.
        - The results can vary depending on the initial center points, making it sensitive to initialization.
            
            (The initial centers are selected randomly, so the results can be influenced by this randomness.)
            
    4. Formula 
        - **Center Calculation**: The mean of all data points within the cluster.
        - Center: $\frac{\text{Sum of data points in the cluster}}{\text{Number of data points}}$
        - Data Point Assignment:
            - Each data point is assigned to the closest center.
            - Assignment is based on distance (Euclidean distance).
        - Distance ($x$, $m$) = $n$ is the number of dimensions, $x_j$ is the $j$-th dimensional value of data point $x$.

2. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    - A **density-based** clustering method that handles **noise**.
    - This algorithm forms clusters based on the density of data points around them.
    
    1. Process
        1. Randomly select a data point.
        2. Find other data points within a certain radius (epsilon) around the selected data point.
        3. If the number of neighboring data points exceeds a minimum number (minPts), assign these data points to the same cluster.
        4. Expand the cluster by recursively adding neighboring data points.
        5. Data points not included in any cluster are treated as **noise**.
        
    2. Advantages
        - It does not assume the shape of clusters, allowing for the identification of clusters in various shapes.
        - There is no need to pre-specify the number of clusters.
        - It can handle noise, allowing for the identification of outliers in the data.
        
    3. Disadvantages
        - The performance can be sensitive to the epsilon (radius) and minPts (minimum points) parameters.
        - Computational costs may increase with large datasets.
        
    4. Formula
        - Uses distance calculations (Euclidean distance).
        - Data points within epsilon distance are considered neighbors.
        - Points not within any cluster are treated as noise.




# Korean

## 클러스터링

클러스터(Cluster): **특성이 비슷한 것**

클러스터링: **특성이 비슷한 것**들을 모으는 것.

### 클러스터링의 목적

- **더 빠른 연산** (매번 모든 데이터의 특성들을 분석해서 분류하고 연산하면 **연산비**가 증가)
- 새로운 특성을 가진 데이터가 생기면 클러스터에 편입.
- 클러스터들을 모아서 더 거대한 클러스터들을 생성.

### 클러스터링 알고리즘

1. $k$-Means 클러스터링 
    - 데이터들을 $k$개의 클러스터로 그룹화하는 알고리즘.
    - 각 클러스터들은 중심을 가지며, 데이터 포인트는 가장 가까운 중심에 할당된다.
    
    1. 동작 과정
        1. 클러스터의 중심점을 임의로 선택($k$가 의미하는 것은 클러스터의 수).
        2. 각 데이터 포인트를 가장 가까운 중심점에 할당.
        3. 클러스터에 속한 데이터 포인트들의 평균을 계산하여 새로운 중심점을 갱신.
        4. 중심점이 더 이상 변경되지 않을 때까지 b, c 과정을 반복 수행.
        
        **중심점**이 더 이상 변경되지 않는다 = **구심점**이 확정되었다 = 다 모였다 = **클러스터** 완성.
        
    2. 장점
        - 단순하면서도 효율적인 알고리즘.
        - 대용량 데이터셋에서 적용이 가능.
        - 클러스터의 수를 지정하기 용이하다.
        
    3. 단점
        - 클러스터의 모양이 **원형**이라고 가정하므로 다른 모양의 클러스터들은 잘 클러스터링하기 어렵다.
        - 초기 중심점의 위치에 따라 결과가 달라질 수 있으므로 초기화 방법에 따른 영향을 받을 수 있다.
            
            (초기 중심점을 랜덤하게 뽑기 때문에 랜덤의 결과에 영향을 받을 수 있다.)
            
    4. 공식 
        - **중심점 계산**: 해당 클러스터에 속한 모든 데이터 포인트의 평균으로 계산.
        - 중심점: $\frac{\text{cluster}}{\text{number of data points}}$
        - 데이터 포인트 할당:
            - 각 데이터 포인트는 가장 가까운 중심점에 할당.
            - 거리를 기반으로 할당 (유클리드 거리).
        - 거리($x$, $m$): $n$은 데이터 포인트의 차원수, $x_j$는 데이터 포인트 $x$의 $j$번째 차원의 값.
            
2. DBSCAN (Density-Based Spatial Clustering of Applications with Noise)
    - **노이즈**를 가진 **밀도 기반** 클러스터링.
    - 데이터 포인트 주변의 밀도를 기반으로 클러스터를 형성하는 알고리즘.
    
    1. 동작 과정
        1. 임의의 데이터 포인트를 선택한다.
        2. 선택된 데이터 포인트에서 일정 반경(epsilon)에 있는 다른 데이터 포인트들을 찾는다.
        3. 이웃 데이터 포인트의 개수가 최소 개수(minPts) 이상이면 해당 데이터 포인트들을 하나의 클러스터로 할당한다.
        4. 이웃 데이터 포인트들도 반복적으로 클러스터에 추가하면서 클러스터를 확장시킨다.
        5. 클러스터에 포함되지 않은 데이터 포인트들은 **노이즈(noise)** 로 처리한다.
        
    2. 장점
        - 클러스터의 모양을 미리 가정하지 않기 때문에 다양한 모양의 클러스터를 찾을 수 있다.
        - 클러스터의 개수를 미리 지정할 필요가 없다.
        - 노이즈를 처리할 수 있어서 데이터에서 이상치를 식별할 수 있다.
        
    3. 단점
        - 성능이 epsilon(반경), minPts(최소 인접 포인트 개수) 값에 따라 민감하게 반응할 수 있다.
        - 대규모 데이터셋 처리 시에 계산 비용이 증가할 수 있다.
        
    4. 공식
        - 거리 계산을 사용 (유클리드 거리).
        - 데이터 포인트 간의 거리가 epsilon 이내인 경우 이웃 데이터 포인트로 판단.
        - 나머지는 노이즈로 처리한다.




