'''
Pytorch란? : 메타(구 페이스북)에서 개발한 파이썬 기반의 오픈소스 딥러닝 프레임워크이다.
유연성이 높고 코드가 직관적이며, GPU를 활용한 빠른 연산과 동적 계산 그래프(Define-by-Run) 방식을 지원한다.

<설치 방법>

1. 구글 코랩 사용하기

2. 로컬에 직접 설치하기
=> Bash : pip install torch torchvision

3. NVIDIA GPU(CUDA) 설치
=> pip install torch torchvision --index-url https://download.pytorch.org/models/*



<설치 후 작동 확인>

Bash :

import torch
print(torch.__version__)
2.11.0+cpu


'''