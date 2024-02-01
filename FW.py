import numpy as np 
import time

class FW:
    def __init__(self,
                 C = 0.01,
                 x = None,
                 epsilon = 1e-5,
                 max_iter = 1e5,):
        self.C = C
        self.x = x
        self.epsilon = epsilon
        self.max_iter = max_iter
        self.A = None
        self.o_f_value = None
        self.history = None

    def _init_x(self,n):
        x_0 = np.zeros((n,1))
        x_0[0] = 1.0
        return x_0
        
    def _add_bias(self,X):
        n = X.shape[0]
        x_with_bias = np.concatenate((X, np.ones((n,1))), axis=1)
        return x_with_bias
    
    def _create_A(self, X, y):
        X = self._add_bias(X)
        Z = X.T * y
        I = (1/ np.sqrt(self.C)) * np.eye(X.shape[0])
        A = np.concatenate((Z, I), axis=0)
        return A
    
    def _objective_function(self, x= None):
        if x is None:
            x = self.x
        return (np.linalg.norm(np.dot(self.A, x), ord=2) ** 2).item()
    
    def _gradient(self,):
        At = self.A.dot(self.x)
        derivative = 2 * self.A.T.dot(At)
        return derivative

    def _LMO(self, grad):
        index = grad.argmin()
        s = np.zeros_like(grad)
        s[index] = 1
        return s
    
    def _fw_step(self):
        x = self.x
        grad = self._gradient()
        s = self._LMO(grad)
        d_fw = s - x
        duality_gap = - np.dot(grad.T, d_fw).item()
        return grad, s, d_fw, duality_gap 
    
    def _armijo(self, d_fw, gap, step_size_max):
        x_prev = self.x
        o_f_prev = self.o_f_value
        alpha, delta = 0.1, 0.8

        step_size = step_size_max
        x_new = x_prev + step_size * d_fw
        o_f_new = self._objective_function(x_new)
        m = 0
        while o_f_new > o_f_prev + alpha * step_size * (-gap):
            m += 1
            step_size = delta * step_size
            x_new = x_prev + step_size * d_fw
            o_f_new = self._objective_function(x_new)
            if m >= 1000:
                break

        return x_new, o_f_new

    def _calc_w_b(self, features):
        wb = self.A.dot(self.x)
        self.w = wb[: features]
        self.b = wb[features: features + 1]

    def predict(self, X):
        if not hasattr(self, 'w') or not hasattr(self, 'b'):
            raise ValueError("""w and b have not been calculated.
            please fit the model first to obtain their optimal values""")
        if not isinstance(X, np.ndarray):
            X = np.array(X)

        y_pred = X.dot(self.w) + self.b
        return np.where(y_pred > 0, 1, -1)
    
    def accuracy(self, y, y_pred= None, X= None):
        if X is None and y_pred is None:
            raise ValueError("One of X or y_pred must be specified")
        elif X is not  None and y_pred is not None:
            raise ValueError("Only one of X or y_pred must be specified")
        elif X is not None:
            y_pred = self.predict(X)
        
        if y.ndim == 1 or y_pred.ndim == 1:
            y = y.reshape(-1, 1)
            y_pred = y_pred.reshape(-1, 1)

        if y.shape != y_pred.shape:
            raise ValueError("y and y_pred must have the same shape")
        return np.sum(y_pred == y) / len(y)


    def fit(self, X, y):
        self.A = self._create_A(X, y)
        n = self.A.shape[1]
        features = X.shape[1]
        if self.x is None:
            self.x = self._init_x(n)
        self.o_f_value = self._objective_function()
        step_size_max = 1

        self.history = {"iteration": [0],
                        "cpu_time": [0],
                        "of_value": [self.o_f_value],
                        "duality_gap": []}
        
        start_time = time.time()
        for iter in range(self.max_iter):
            grad, s, d_fw, duality_gap = self._fw_step()
            self.history["duality_gap"].append(duality_gap)
            if duality_gap <= self.epsilon:
                print(f"Duality Gap <= Epsilon\nEnd of training at iteration {iter+1}.")
                break
            
            self.x, self.o_f_value = self._armijo(d_fw, duality_gap, step_size_max)

            end_time = time.time()
            self.history["iteration"].append(iter+1)
            self.history["of_value"].append(self.o_f_value)
            self.history["cpu_time"].append(end_time - start_time)

        self._calc_w_b(features)



class ASFW(FW):
    def _away_step(self, grad, duality_gap, d_fw):
        v = np.zeros(grad.shape)
        active_alphas = np.where(self.x > 0)[0]
        potential_ascent = grad[active_alphas]
        index_grad_max = potential_ascent.argmax()
        v_index = active_alphas[index_grad_max]
        v[v_index] = 1
        d_a = self.x - v
        gap_away = - np.dot(grad.T, d_a).item()

        if duality_gap >= gap_away:
            d = d_fw
            step_size_max = 1
            gap = duality_gap
        else:
            d = d_a
            alpha_v = self.x[v_index]
            step_size_max = alpha_v / (1 - alpha_v)
            gap = gap_away
        return d, gap, step_size_max

        
    def fit(self, X, y):
        self.A = self._create_A(X,y)
        n = self.A.shape[1]
        features = X.shape[1]
        if self.x is None:
            self.x = self._init_x(n)

        self.o_f_value = self._objective_function()

        self.history = {
            "iteration": [0],
            "cpu_time": [0],
            "of_value": [self.o_f_value],
            "duality_gap": []}


        start_time = time.time()
        for iter in range(self.max_iter):
            grad, s, d_fw, duality_gap = self._fw_step()
            self.history['duality_gap'].append(duality_gap)
            if duality_gap <= self.epsilon:
                print(f"Duality Gap <= Epsilon\nEnd of training at iteration {iter+1}.")
                break
                
            d, gap, step_size_max = self._away_step(grad, duality_gap, d_fw)
            self.x, self.o_f_value = self._armijo(d, gap, step_size_max)

            end_time = time.time()
            self.history["iteration"].append(iter+1)
            self.history["of_value"].append(self.o_f_value)
            self.history["cpu_time"].append(end_time - start_time)

        self._calc_w_b(features)
            


class PWFW(FW):
    def _pairwise_step(self, grad, s):
        v = np.zeros(grad.shape)
        active_alphas = np.where(self.x > 0)[0]
        potential_ascent = grad[active_alphas]
        index_grad_max = potential_ascent.argmax()
        v_index = active_alphas[index_grad_max]
        v[v_index] = 1
        d_pw = s - v
        step_size_max = self.x[v_index]
        gap_pw = - np.dot(grad.T, d_pw).item()
        return d_pw, gap_pw, step_size_max 

    
    def fit(self, X, y):
        self.A = self._create_A(X,y)
        n = self.A.shape[1]
        features = X.shape[1]
        if self.x is None:
            self.x = self._init_x(n)

        self.o_f_value = self._objective_function()

        self.history = {
            "iteration": [0],
            "cpu_time": [0],
            "of_value": [self.o_f_value],
            "duality_gap": []}

        start_time = time.time()
        for iter in range(self.max_iter):
            grad, s, d_fw, duality_gap = self._fw_step()
            self.history['duality_gap'].append(duality_gap)
            if duality_gap <= self.epsilon:
                print(f"Duality Gap <= Epsilon\nEnd of training at iteration {iter+1}.")
                break

            d, gap_pw, step_size_max = self._pairwise_step(grad, s)
            self.x, self.o_f_value = self._armijo(d, gap_pw, step_size_max)

            end_time = time.time()
            self.history["iteration"].append(iter+1)
            self.history["of_value"].append(self.o_f_value)
            self.history["cpu_time"].append(end_time - start_time)

        self._calc_w_b(features)
