# 하이퍼파라미터 : 컴퓨터가 스스로 결정할 수 없고, 학습을 시작하기 전에 사람이 수동으로 정해줘야 하는 설정값들
# => 각 층의 뉴런 수, 배치 크기, 에폭 크기, 매개변수 갱신 시의 학습률과 감중치 감소 등

"""
하이퍼 파라미터의 성능을 평가할 때는 시험 데이터를 사용하면 안됨.
=> 하이퍼 파라미터 값이 시험 데이터에 과적합되기 때문

따라서 하이퍼 파라미터를 조정할 대는 하이퍼 파라미터 전용의 확인 데이터가 필요함
=> 그것이 바로 "검증 데이터(Validation Data)"

훈련 데이터 : 모델이 학습하며 가중치(W,b)를 학습 (전체 데이터의 60~80%)
검증 데이터 : 훈련 데이터로 학습을 마친 모델에게, 현재 세팅한 하이퍼 파라미터가 맞는지 중간 점검 (전체 데이터의 10~20%)
시험 데이터 : 최종 모델의 범용 성능 평가
"""
from tensorflow.python.ops.gen_dataset_ops import shuffle_dataset

from mnist import dataset_dir, load_mnist

# 검증 데이터를 얻기 위해 훈련 데이터의 20%를 떼어내는 코드

(x_train, t_train), (x_test, y_test) = load_mnist()

# 훈련 데이터를 섞는다
x_train, t_train = shuffle_dataset(x_train, t_train)

# 훈련 데이터 비율을 정한다
validation_rate = 0.20

# 정확히 몇 장인지 계산한다
validation_num = int(x_train.shape[0] * validation_rate)

# 데이터 슬라이싱
# 검증 데이터 12,000장 확보(앞에서부터 자르기)
x_val = x_train[:validation_num]
t_val = t_train[:validation_num]

# 남은 훈련 데이터 48,000장 확보(나머지 뒷부분 전부)
x_train = x_train[validation_num:]
t_train = t_train[validation_num:]
