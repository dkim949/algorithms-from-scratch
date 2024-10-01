import sys
import os
import logging
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

logging.basicConfig(level=logging.INFO)

# 프로젝트 루트 디렉토리를 Python 경로에 추가
sys.path.append(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)

from sklearn.preprocessing import StandardScaler
from utils.data_loader import load_data

data = load_data("california_housing")
logging.info(f"Data loaded: {data.head()}")
logging.info(f"Data columns: {data.columns}")

# 'MedInc' (중간 소득) 특성을 선택하여 단순 선형 회귀 수행
X = data["MedInc"].values.reshape(-1, 1)
y = data["PRICE"].values.reshape(-1, 1)

# 데이터 정규화
scaler_X = StandardScaler()
scaler_y = StandardScaler()
X_scaled = scaler_X.fit_transform(X)
y_scaled = scaler_y.fit_transform(y)
logging.info(f"Data scaled")


# Gradient Descent 함수
def gradient_descent(X, y, learning_rate=0.01, n_iterations=100):
    m = X.shape[0]
    theta = np.random.randn(2, 1)  # 모델 파라미터(가중치와 편향)를 무작위로 초기화
    X_b = np.c_[np.ones((m, 1)), X]  # 입력 데이터에 편향 항(1)을 추가

    theta_history = [theta.copy()]  # 파라미터 업데이트 기록을 저장할 리스트

    for iteration in range(n_iterations):
        gradients = (
            2 / m * X_b.T.dot(X_b.dot(theta) - y)
        )  # 오차에 대한 그래디언트를 계산
        theta = theta - learning_rate * gradients  # 파라미터를 업데이트
        theta_history.append(theta.copy())  # 파라미터 업데이트 기록

    return np.array(theta_history)


# 그래디언트 디센트 실행
theta_history = gradient_descent(
    X_scaled, y_scaled, learning_rate=0.1, n_iterations=200
)

# 시각화
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(X_scaled, y_scaled, color="b", alpha=0.5)
(line,) = ax.plot([], [], "r-", lw=2)
iteration_text = ax.text(0.02, 0.95, "", transform=ax.transAxes)

ax.set_xlim(X_scaled.min(), X_scaled.max())
ax.set_ylim(y_scaled.min(), y_scaled.max())
ax.set_xlabel("Normalized Median Income")
ax.set_ylabel("Normalized Median House Value")
ax.set_title("California Housing: Income vs House Value - Gradient Descent")


def init():
    line.set_data([], [])
    iteration_text.set_text("")
    return line, iteration_text


def animate(i):
    # if i % 2 == 0:
    theta = theta_history[i]
    X_plot = np.array([[X_scaled.min()], [X_scaled.max()]])
    X_plot_b = np.c_[np.ones((2, 1)), X_plot]
    y_plot = X_plot_b.dot(theta)
    line.set_data(X_plot, y_plot)
    iteration_text.set_text(f"Iteration: {i}")
    return line, iteration_text


# 애니메이션 생성
anim = FuncAnimation(
    fig,
    animate,
    init_func=init,
    frames=len(theta_history),
    interval=50,
    blit=True,
    repeat=False,
)

plt.show()

# 최종 결과 출력
final_theta = theta_history[-1]
print("최종 모델 (정규화된 스케일):")
print(f"y = {final_theta[1][0]:.4f}x + {final_theta[0][0]:.4f}")

# 원래 스케일로 변환
original_slope = final_theta[1][0] * (scaler_y.scale_ / scaler_X.scale_)
original_intercept = (
    (final_theta[0][0] * scaler_y.scale_)
    + scaler_y.mean_
    - (original_slope * scaler_X.mean_)
)

# numpy 배열에서 스칼라 값으로 변환
original_slope = float(original_slope)
original_intercept = float(original_intercept)

print("\n원래 스케일의 모델:")
print(f"집값 = {original_slope:.2f} * 중간소득 + {original_intercept:.2f}")
