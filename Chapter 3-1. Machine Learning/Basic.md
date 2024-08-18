# English

## Basic of Machine Learning

### Machine Learning

Machine learning refers to the process of teaching machines to learn from data. Initially, machines were trained to perform tasks that could replace human labor, but today, they are capable of learning on their own.

- **By statistically inferring patterns and trends from existing data or data streams**, machines can predict and analyze future data or trends.
- Input: Existing data
- Output: Future data
- The goal is to enable machines to **learn on their own** through experience.

### Terminology

**Big Data**

- A field that includes the techniques and methodologies for handling large volumes of data.
- Involves the collection, storage, processing, and analysis of data to extract useful information.

**Artificial Intelligence (AI)**

- A field focused on designing computer programs that exhibit intelligence similar to human beings.
- AI technologies are used in various fields such as learning, reasoning, natural language processing, and more.

**Machine Learning**

- A field where computers are equipped with the ability to learn from data, discover patterns, make predictions, and make decisions autonomously.
- Develops algorithms that automatically improve through data and experience.

### Types of Machine Learning

**Supervised Learning**

- A learning approach where the goal is to match predefined correct answers.
- Classification: Choosing one option from several.
    - Example: Spam email classification program
      
      → The goal is to read an email and determine if it is spam. There is a correct answer. 

- Regression: Predicting outcomes when there are countless and continuous possible results.

    Predicts future data by finding patterns in existing data.

    - Example: Predicting apartment prices, stock prices, etc.

**Unsupervised Learning**

- A learning approach where there is no predefined correct answer, and the goal is to find the answer.
    - Example: Grouping similar news articles together.
      
      Articles can be grouped by date, category, reporter, media outlet, etc.

**Reinforcement Learning**

- A method where AI learns and makes decisions autonomously through experience.
- The AI experiments and learns the optimal choices through trial and error.
- An agent interacts with its environment and learns the optimal policy to maximize rewards.
- Reinforcement learning, such as in deep learning models like ChatGPT, focuses on the concept of "reward."
- Focus on **rewards** when thinking about reinforcement learning.

### Advantages and Disadvantages of Machine Learning

**Advantages**

- Identifies data trends and patterns that humans might miss or overlook.
- Can operate without human intervention once set up.
- Capable of processing various data in dynamic, large-scale, and more complex data environments.

**Disadvantages**

- The initial training phase requires significant costs and time (economic viability of infrastructure).
- It is difficult to accurately interpret results and eliminate uncertainties without expert assistance.

### Steps in Machine Learning

Problem Definition > Data Collection > Data Preprocessing > Feature Selection > Feature Engineering > Model Selection > Model Training > Model Evaluation > Model Improvement > Model Deployment > Report Writing > Code Optimization and Refactoring

### Supervised Learning

- Classification
- Regression

### Unsupervised Learning

- Clustering: Grouping data according to certain rules.
- Dimensionality Reduction:
  
  → Compressing high-dimensional data into lower-dimensional data to visualize it or use it in other analysis tasks.

- Association Rule Learning:

  Discovering relationships or patterns among items in a dataset.

### Reinforcement Learning

- Agent: The entity that performs the learning.
- Environment: The external environment with which the agent interacts and processes rewards.
- State: The state of the environment at a specific moment through interactions between the agent and the environment.
- Action: The set of possible actions an agent can take in a specific state.
- Reward: The reward signal (data, value) the agent receives as a result of performing a specific action.
- **Q-Learning**:
  
  A method where the agent interacts with the environment and learns to find the optimal **Q-function (a function that justifies increasing accuracy)** to select the optimal action.

- **SARSA**: A method where the agent repeatedly performs actions under specific conditions and receives rewards.
  
  State-Action-Reward-State-Action-Reward-State-Action-...


# Korean

## 머신러닝의 기초

### 머신러닝

기계를 학습시키는 것, 기계가 학습하는 것을 의미한다. 처음 단계에서는 기계를 학습시켜, 인간을 대체했지만, 현재는 직접 학습하고 있다.

- **기존 데이터나 데이터의 흐름**을 머신을 통해 통계적으로 추론해서 미래 데이터나 데이터의 흐름을 예측/분석한다.
- Input: 기존 데이터
- Output: 미래 데이터
- 머신이 경험을 통해서 **스스로 학습**하도록 함

### 용어 정리

**빅데이터**

- 대량의 데이터를 다루는 기술과 방법론을 포함하는 분야
- 데이터를 수집하고 저장하고 처리하고 분석하고 정보를 추출하는 과정

**인공지능**

- 사람과 유사한 지능을 갖도록 컴퓨터 프로그램이 설계되는 분야이다.
- 학습, 추론, 자연어 처리, 등등 수많은 분야에서 인공지능 기술을 사용한다.

**머신러닝**

- 컴퓨터가 데이터로부터 스스로 학습하고, 패턴을 발견하고 예측하고 결정하는 능력을 갖추도록 한 분야
- 데이터와 경험을 통해 자동적으로 개선되는 알고리즘을 개발하는 기술

### 머신러닝의 학습 방법

**지도 학습 (Supervised Learning)**

- 답이 정해져 있고, 이 답을 맞추는 것을 목표로 하는 학습이다.
- 분류(Classification): 몇 가지 옵션 중에서 하나를 고르는 것
    - 예시) 스팸 메일 분류 프로그램
        
        → 메일 내용을 읽고, 스팸인지 아닌지 맞추려고 한다. 답이 있다.
        
- 회귀 (Regression): 결과 값이 무수히 많고 연속적일 때 결과 값을 도출
    
    현재까지의 데이터들의 패턴을 찾아서 미래의 데이터를 예측
    
    - 예시) 아파트 가격 예측, 주가 예측 등

**비지도 학습 (Unsupervised Learning)**

- 답이 없는데 이 답을 맞추는 것이 학습의 목표이다.
    - 예시) 기사를 비슷한 기사들끼리 묶어야 한다.
        
        날짜별, 카테고리별, 기자별, 언론사별… 여러 기준으로 분류 가능
        

**강화 학습 (Reinforcement Learning)**

- 인공지능이 스스로 경험을 통해서 배우고 의사 결정을 내리는 방법
- 스스로 실험과 시행착오를 통해 최적의 선택을 학습
- 에이전트가 환경과 상호작용하면서 특정 작업을 수행하고 **보상**을 최대화하기 위한 최적의 정책을 학습
- 대표적으로 딥러닝: ChatGPT 같은 경우 같은 프롬프트를 반복하면 보상 점수를 다운시킨다.
- 강화 학습에서는 “**보상**”에 집중해서 생각해야 한다.

### 머신러닝의 장단점

**장점**

- 사람이 놓치거나 실수할 수 있는 데이터 추세와 패턴을 식별 가능
- 설정 후 사람의 개입 없이도 작업 가능
- 동적이고 대용량이며 더 복잡한 데이터 환경에서 다양한 데이터 처리가 가능

**단점**

- 초기 훈련 시 비용과 시간이 많이 소요된다. (인프라 구축의 경제성)
- 전문가 도움 없이 결과를 정확하게 해석하고 불확실성을 없애기 어려움

### 머신 러닝 진행 순서

문제 정의 > 데이터 수집 > 데이터 전처리 > 특성 선택 > 특성 엔지니어링 > 모델 선택 > 모델 학습 > 모델 평가 > 모델 개선 > 모델 배포 > 보고서 작성 > 코드의 최적화 및 리팩토링

### 지도 학습

- 분류
- 회귀

### 비지도 학습

- 클러스터링 (Clustering) : 전달된 데이터를 어떤 일정한 규칙에 따라서 묶는 것
- 차원 축소 (Dimensionality Reduction)
    
    → 고차원 데이터의 특성을 저차원 데이터의 특성으로 압축해서 데이터를 시각화하거나 다른 분석 작업에 활용하는 데 사용
    
- 연관규칙 학습 (Association Rule Learning)
    
    데이터셋에서 항목들 사이의 관계나 패턴을 찾는 작업
    

### 강화 학습

- 에이전트(Agent) : 학습을 수행하는 주체
- 환경 (Environment) : 에이전트가 상호작용하는 외부 환경, 보상에 대한 처리
- 상태(State) : 에이전트와 환경의 상호작용을 통해 특정 시점에서의 환경의 상태
- 행동(Action) : 에이전트가 특정 상태에서 취할 수 있는 선택 가능한 행동의 집합
- 보상(Reward): 에이전트가 특정 행동을 수행한 결과 받는 보상 신호(데이터, 값)
- **Q-Learning**
    
    에이전트가 환경과 상호작용하면서 최적의 **Q-함수(정확도를 높이는 근거 함수)**를 찾아가면서 최적의 행동을 선택하도록 학습하는 방법
    
- **SARSA:** 특정 상태하에 하는 행동에 따른 보상의 반복
    
    상태-행동-보상-상태-행동-보상-상태-행동-….



