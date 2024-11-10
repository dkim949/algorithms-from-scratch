import numpy as np

class MLP:
    def __init__(self, layer_sizes):
        """
        layer_sizes: 각 층의 뉴런 수를 담은 리스트
        예: [4, 6, 3, 2] -> 입력층 4개, 첫 번째 은닉층 6개, 두 번째 은닉층 3개, 출력층 2개
        """
        self.layer_sizes = layer_sizes
        self.n_layers = len(layer_sizes) - 1
        
        # 가중치와 편향 초기화
        self.weights = []
        self.biases = []
        
        for i in range(self.n_layers):
            # He 초기화 사용
            w = np.random.randn(layer_sizes[i], layer_sizes[i+1]) * np.sqrt(2.0/layer_sizes[i])
            b = np.zeros(layer_sizes[i+1])
            
            self.weights.append(w)
            self.biases.append(b)
    
    def tanh(self, x):
        return np.tanh(x)
    
    def tanh_derivative(self, x):
        return 1 - np.tanh(x)**2
    
    def forward(self, X):
        """순전파 수행"""
        self.activations = [X]  # 각 층의 활성화값 저장
        self.z_values = []      # 활성화 함수 적용 전의 값 저장
        
        A = X
        for i in range(self.n_layers):
            Z = np.dot(A, self.weights[i]) + self.biases[i]
            self.z_values.append(Z)
            
            A = self.tanh(Z)
            self.activations.append(A)
        
        return A
    
    def backward(self, X, y, learning_rate=0.01):
        """역전파 수행"""
        m = X.shape[0]  # 배치 크기
        
        # 출력층에서의 오차
        dA = 2 * (self.activations[-1] - y) / m  # MSE 손실의 미분
        
        for i in reversed(range(self.n_layers)):
            dZ = dA * self.tanh_derivative(self.z_values[i])
            
            # 가중치와 편향에 대한 그래디언트 계산
            dW = np.dot(self.activations[i].T, dZ)
            dB = np.sum(dZ, axis=0)
            
            # 이전 층으로의 그래디언트
            if i > 0:
                dA = np.dot(dZ, self.weights[i].T)
            
            # 가중치와 편향 업데이트
            self.weights[i] -= learning_rate * dW
            self.biases[i] -= learning_rate * dB
    
    def train(self, X, y, epochs=1000, learning_rate=0.01, verbose=True):
        """모델 학습"""
        losses = []
        weights_history = []  # 첫 번째 층의 첫 번째 가중치 추적  (시각화를 위해)
        
        for epoch in range(epochs):
            # 순전파
            y_pred = self.forward(X)
            
            # 손실 계산
            loss = np.mean((y_pred - y) ** 2)
            losses.append(loss)
            
            # 특정 가중치 추적 (예: 첫 번째 층의 첫 번째 가중치)  (시각화를 위해)
            weights_history.append(self.weights[0][0, 0])
            
            # 역전파
            self.backward(X, y, learning_rate)
            
            # 진행상황 출력
            if verbose and epoch % 100 == 0:
                print(f"Epoch {epoch}, Loss: {loss:.6f}, Weight: {self.weights[0][0, 0]:.6f}")
        
        return losses, weights_history
    
    def predict(self, X):
        """예측 수행"""
        return self.forward(X)