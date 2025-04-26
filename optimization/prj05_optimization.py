import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 스타일 설정
sns.set(style='whitegrid')
plt.rcParams['figure.figsize'] = [10, 8]

# 1. 간단한 2차원 함수 정의
def quadratic_function(x, y):
    return x**2 + y**2

def gradient_quadratic(x, y):
    return np.array([2*x, 2*y])

# 2. 등고선 시각화를 위한 데이터 생성
x = np.linspace(-5, 5, 100)
y = np.linspace(-5, 5, 100)
X, Y = np.meshgrid(x, y)
Z = quadratic_function(X, Y)

# 3. 등고선 플롯 생성
plt.figure(figsize=(10, 8))
contour = plt.contour(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar(contour)
plt.xlabel('X')
plt.ylabel('Y')
plt.title('2차원 이차함수의 등고선')
plt.show()

# 4. 경사하강법 구현
def gradient_descent(start_point, learning_rate, num_iterations):
    points = [start_point]
    current_point = start_point
    
    for _ in range(num_iterations):
        grad = gradient_quadratic(current_point[0], current_point[1])
        current_point = current_point - learning_rate * grad
        points.append(current_point)
    
    return np.array(points)

# 5. 경사하강법 실행 및 시각화
start_point = np.array([4.0, 4.0])
learning_rate = 0.1
num_iterations = 50

points = gradient_descent(start_point, learning_rate, num_iterations)

# 등고선과 경사하강법 경로 시각화
plt.figure(figsize=(10, 8))
contour = plt.contour(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar(contour)
plt.plot(points[:, 0], points[:, 1], 'r-', lw=2, label='경사하강법 경로')
plt.plot(points[0, 0], points[0, 1], 'go', label='시작점')
plt.plot(points[-1, 0], points[-1, 1], 'ro', label='종료점')
plt.xlabel('X')
plt.ylabel('Y')
plt.title('경사하강법 진행 경로')
plt.legend()
plt.show()

# 6. 학습률에 따른 경사하강법 비교
learning_rates = [0.01, 0.1, 0.5, 1.0]
num_iterations = 50

plt.figure(figsize=(15, 12))
for i, lr in enumerate(learning_rates):
    points = gradient_descent(start_point, lr, num_iterations)
    
    plt.subplot(2, 2, i+1)
    contour = plt.contour(X, Y, Z, levels=20, cmap='viridis')
    plt.plot(points[:, 0], points[:, 1], 'r-', lw=2)
    plt.plot(points[0, 0], points[0, 1], 'go')
    plt.plot(points[-1, 0], points[-1, 1], 'ro')
    plt.title(f'학습률 = {lr}')
    plt.xlabel('X')
    plt.ylabel('Y')

plt.tight_layout()
plt.show() 