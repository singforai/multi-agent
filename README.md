# multi-agent

가장 key idea 
  1. permutation invariant(= 입력 벡터 요소의 순서와 상관없이 같은 출력을 생성하는 모델)를 이용한 joint - V-value 연산(share_obs 사용하지 않고)
  2. self-attention(= 에이전트간의 완전한 정보 교환)을 통한 permutation equivalant action sampling

baseline
  1. MAT
     mat는 transformer를 통해 marl문제를 sequential problem으로 변환함으로써 search space가 에이전트에 대해 선형적이도록 유도했다. 다만 에이전트의 입력 순서에 따라 에이전트의 행동 집합이 크게 달라질 수 있다는 점은 mat의 행동 선택 과정이 permutation equivalance하지 않다는 것을 보여준다. positional encoding을 사용하지 않았다.(20p) 훈련 과정에서 다양한 에이전트의 순열로 훈련해야 함으로 훈련 시간이 매우 증가한다. 또한 mat의 encoder에서 V value를 출력할 땐 모든 에이전트의 v가 따로 계산된다. 
     기존 MAPPO 계열 알고리즘은 actor 네트워크에서 각 에이전트간에 정보교환이 일어나지 않는다. 이것은 permutation equivalance이지만 에이전트의 협력적 행동을 유도하는데 어려움을 느끼게 할 수 있다. 단 critic에서 share observation을 통해 공통된 v-value를 추정해낼 수 있지만 share observation이 반드시 존재해야 한다는 제약조건이 존재하며 이것은 현실 세계에서의 적용을 어렵게 만든다. 

our model 
보통 CNN과 RNN이 permutation invariant하지 못하다고 불리는 이유는 각 픽셀, token을 입력 단위로 사용하기 때문, 하지만 우리가 주장하는 permutation invariant란 에이전트간의 입력 순서에 대한 invariant이기에 각 에이전트가 sequential한 데이터를 훈련한다고 해도 문제가 되지 않는다. 따라서 각 에이전트에 대해 독립적으로 연산되는 MLP, RNN을 통해 embedding한 정보를 self-attention block을 이용해 정보를 교환한 뒤, 각각 action을 샘플링하고 joint-V-value를 계산한다. 

ablation research 
  1. num head 개수에 따른 성능 변화
  2. num seed vector 개수에 따른 성능 변화????
  3. pma block 유무에 따른 성능 차이(joint-V value를 계산해낼 수 있는가?)
  4. actor / critic 역할하는 네트워크를 완전히 분리해서 실험해보기 => 굳이 transformer 구조를 취할 필요가 있는지에 대한 답이 필요함.
  5. 학습이 끝난 모델의 attention head 가중치를 분석해보는 것이 좋을까?

Quset
  1. MAT의 인코더에서 self-attention을 통해 계산된 각 에이전트의 V값들은 joint-advantange를 계산하기 위해 사용된다. 그렇다면 joint v-value는 어떻게 계산되는 것이며(각 에이전트의 v-value는 따로 계산되는 것일까?) 이것은 permutation invariant하다고 말할 수 없으니 문제가 생길 수 있음.(한 state에서의 고유한 v를 제대로 추측해내지 못한다는 것)
  2. num seed vector는 1개 이상이 되더라도 permutation invariant에 영향을 미치지 않는다?
  3. joint-V value만 계산하면 각 에이전트의 행동 변화를 곱으로 표현하지 않아도(jrpo)팀을 위한 action을 취하는 것이 가능하지 않나??
  4. state value를 하나로 통일한 것이 critic의 search space를 줄였다고 할 수 있을까? 만약 그렇다면 이것은 수식적으로 표현할 수 있을까?
  5. share observation을 쓰지 않고도 central v value를 추정할 수 있는 것이 어떤 장점이 될 수 있을까?
  6. 그냥 각 에이전트의 임베딩 벡터를 더하거나 평균내고 그걸로 joint - V value를 구하는 것과 set transformer가 어떤 차이가 있을까? 
