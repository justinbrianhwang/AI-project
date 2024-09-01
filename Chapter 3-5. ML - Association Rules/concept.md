# English

![Association Rules](https://annalyzin.files.wordpress.com/2016/04/association-rules-network-graph2.png)

The image above illustrates how to identify relationships between different items. This concept is similar to a recommendation system. For example, when you purchase an item on an online store, the system often suggests related products. This algorithm is known as association rules. We will explore this principle in more detail.

## Association Rules Learning

**Association**: Refers to the relationship between items.

**Association Rule**: A rule that defines the relationship between items.

**Association Rule Learning**: In machine learning, it refers to the process of teaching a system to identify these association rules.

In simple terms, an association rule is a guideline that defines the relationship between events (transactions) that occur simultaneously.

### (1) Terms Related to Association Rules

1. **Itemset**: A collection of items.
2. **$k$-itemset**: Refers to an itemset that contains $k$ items.
   - Example: 2-itemset: {diapers:beer}
3. **Support**: The ratio of transactions in which two items appear together out of the total number of transactions.
   - $\text{Counting support for a set of items} \over \text{Total number of transaction data}$
   - Example: {diapers:milk}, {diapers:milk:beer}, {milk:beer}, {diapers:milk:beer}, {diapers}
   - Out of 5 total transactions, the support for the itemset {diapers:milk} is:
   - $= {3\over5 }= 0.6$
4. **Support count**: The number of transactions in which an itemset appears (in the example above, this count is 3).
   - $={ \text{Support count} \over \text{Number of transactions}}$
5. **Frequent itemset**: An itemset that meets or exceeds a specified support threshold.
6. **Confidence**: The ratio of transactions where both X and Y appear, given that X has appeared.
   - Example: The confidence that ‘diapers’ are purchased together with ‘milk’ when ‘diapers’ are purchased.
   - In simpler terms, the likelihood of buying milk when diapers are purchased.
7. **Lift**: The ratio of the observed support of X and Y appearing together to the expected support if X and Y were independent.
   - If the lift is greater than 1, it indicates a positive correlation. A lift of 1 indicates independence, and less than 1 indicates a negative correlation.

### (2) Shopping Data Example

- Transaction ID: T1  
  - Purchased items: bread, milk
- Transaction ID: T2  
  - Purchased items: diapers, eggs, beer, bread
- Transaction ID: T3  
  - Purchased items: diapers, beer, milk, cola
- Transaction ID: T4  
  - Purchased items: diapers, beer, bread, milk
- Transaction ID: T5  
  - Purchased items: diapers, bread, milk, cola

It would be difficult for a human to find the relationships within these transactions manually. Although this dataset is small, in real-world scenarios, datasets can contain thousands or even millions of records, necessitating the use of machine learning to identify patterns.

### (3) Apriori Algorithm

1. A popular algorithm for learning association rules.
2. It selects frequent itemsets and generates association rules with high confidence.
3. Frequent itemsets are generated progressively.

### (4) Applications of Association Rule Learning

Association rule learning is commonly used in recommendation systems, such as analyzing consumer purchasing behavior, product recommendations, credit card fraud detection, and web user behavior analysis.

### (5) Main Steps in Implementing Association Rule Learning

1. **Data Collection**: Gather transaction data, and log data to create itemsets.
2. **Data Preprocessing**: Remove duplicates, clean unnecessary data, and organize data into transaction formats.
3. **Itemset Generation**: Identify which items frequently occur together in transactions to understand relationships.
4. **Association Rule Mining**:
   - Use algorithms like Apriori to find association rules.
   - Calculate associations based on frequency.
   - Remove irrelevant rules.
   - Highlight important rules through post-processing.
5. **Rule Evaluation**: Use metrics like support, confidence, and lift to evaluate the rules.
6. **Result Interpretation and Application**: Interpret the discovered rules and apply them in practical scenarios.
7. **Periodic Updates**: Update the rules periodically as new data is added or existing data changes to maintain the relevance of the associations.



# Korean

![Association Rules](https://annalyzin.files.wordpress.com/2016/04/association-rules-network-graph2.png)

위의 그림을 보면 서로 어떤 관계가 있는지 보는 것이다. 추천 시스템을 생각하면 쉽다. 
쇼핑몰에서 한 물건을 사면 그와 유사한 상품을 노출시키곤 한다. 이러한 알고리즘을 연관규칙이라 한다. 이 원리를 다룰 것이다. 

## Association Rules Learning

**연관** : 서로 관련이 있음

**연관 규칙** : 연관에 대한 규칙을 의미한다. 

**연관 규칙 학습** : 머신 러닝에서 연관 규칙을 학습 시키는 것을 의미한다. 

연관 규칙을 한마디로 정의하면, 동시에 발생한 사건(Transaction)들 간의 관계를 정의한 규칙 

### (1) 연관규칙 관련 용어 

1. **itemset(항목집합)** : 항목들을 모음 
2. **$k$-itemset($k$-항목집합)** : $k$개의 항목집합을 의미한다.  
   - 예시: 2-itemset : {기저귀:맥주}
3. **support(지지도)** : 전체 경우의 수 중에서 두 아이템이 같이 나오는 비율  
   - $\text{Counting support for a set of items} \over \text{Total number of transaction data}$
   - 예시: {기저귀:우유}, {기저귀:우유:맥주}, {우유:맥주}, {기저귀:우유:맥주}, {기저귀}  
   - 전체 경우의 수 (5개) 중에서 두 아이템 ({기저귀:우유})가 같이 나오는 비율  
   - $= {3\over5 }= 0.6$
4. **support count(지지회수)** : 부분항목집합이 나타난 거래의 수 (위의 예에서는 3)  
   - $={ \text{Support count} \over \text{Number of transactions}}$
5. **빈발 항목집합** : 지정된 지지도 이상의 지지도를 갖는 항목 집합 
6. **신뢰도** : X가 나온 경우 중에서 X와 Y가 함께 나오는 비율  
   - 예시: ‘기저귀’가 나왔을 때 ‘기저귀, 우유’가 나올 비율  
   - 쉽게 말해, 기저귀를 샀을 때 우유도 같이 살 비율
7. **lift(향상도)** : 동시에 두 사건이 발생한 비율 X와 Y가 같이 나오는 비율을 X가 나올 비율과 Y가 나올 비율의 곱으로 나눈 값  
   - 향상도가 1보다 높을 때, positively correlation, 1일 때 independent, 1 미만일 때 negatively correlation이라 한다. 

### (2) 쇼핑 데이터 예시 

- 거래 번호: T1  
  - 구매항목: 빵, 우유
- 거래 번호: T2  
  - 구매항목: 기저귀, 달걀, 맥주, 빵
- 거래 번호: T3  
  - 구매항목: 기저귀,맥주, 우유, 콜라
- 거래 번호: T4  
  - 구매항목: 기저귀, 맥주, 빵, 우유
- 거래 번호: T5  
  - 구매항목: 기저귀, 빵, 우유, 콜라

위의 상관 관계를 사람은 할 수 없다. 위의 자료는 5개지만, 실제 데이터는 몇천, 몇만 개이기에 기계를 학습시키는 것이다. 

### (3) apriori 알고리즘 

1. 대표적인 연관규칙학습 알고리즘 
2. 빈발항목들을 선택하고 그중 신뢰도(confidence)가 높은 연관규칙 생성 
3. 빈발 항목집합들을 점진적으로 생성 

### (4) 연관규칙학습 사용 예시 

소비자 구매 행동 패턴 분석, 상품 추천, 신용카드 사기 탐지, 웹 사용자 행동 분석 등등 추천 시스템에서 주로 사용된다. 

### (5) 연관규칙학습 구현 주요 단계

1. **데이터 수집** : 거래 데이터, 로그 데이터 수집, 항목들의 집합을 구성 
2. **데이터 전처리** : 중복된 항목 제거, 불필요한 데이터 정리, 데이터들을 트랜잭션 형태로 정리 
3. **항목 집합 생성**: 각 트랜잭션에서 어떤 항목들이 함께 발생하는지 파악, 연관성 파악 
4. **연관 규칙 탐색**:  
   - Apriori 알고리즘 등을 활용해 연관규칙 탐색 
   - 빈도수 기반으로 연관성을 계산 
   - 불필요한 규칙 제거 
   - 중요한 규칙을 강조하기 위한 후처리 작업 수행 
5. **규칙평가** : 지지도, 신뢰도, 향상도 등의 지표를 사용해서 규칙을 평가 
6. **결과 해석 및 활용**: 찾아낸 연관규칙을 해석, 실제 업무에 적용
7. **주기적 업데이트** : 데이터가 변경 또는 추가될 때 주기적으로 업데이트  
   - 연관성을 최신 상태로 유지



