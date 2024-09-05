# multi-agent

set transformer를 이용해 share observation을 사용하지 않는 것은 하나의 거대한 장점. 

2024-09-05 
mast-v1 실험 결과
head  수 4개 
seed vector 4개 

3m에서는 rmappo 대비 살짝 뒤지나 MMM에서는 매우 우수한 성능 달성, 하지만 3s5z_vs_3s6z 환경에서는 보상을 높이는 방법을 탐색하지 못함, critic 출력이 actor에 영향을 받는 것이 역전파에서 문제를 일으킬 수도 있는 것으로 보임, 아니면 네트워크의 크기가 너무 작은 것도 문제가 될 수 있음. 