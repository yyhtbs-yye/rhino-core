import math

class SigmoidFadeIn:
    """0 -> 1 S-shaped logistic ramp starting at start_step and ending ~at start_step+duration.
    'eps' controls how close to 0/1 the endpoints get at the boundaries.
    """
    def __init__(self, max_weight=1.0, start_step=5_000, duration=50_000, step_size=5, eps=1e-6):
        assert duration > 0 and step_size >= 1 and 0 < eps < 0.5
        self.start_step = start_step
        self.duration = duration
        self.step_size = max(1, step_size)
        # Choose slope so that at the boundaries (t=0,1) we’re at ~eps and 1-eps:
        self._a = 1.0 * math.log(1.0/eps - 1.0)
        self.max_weight = max_weight

    def __call__(self, step: int) -> float:
        if step < self.start_step:
            return 0.0 * self.max_weight
        k = (step - self.start_step) // self.step_size
        N = max(1, self.duration // self.step_size)
        if k >= N:
            return 1.0 * self.max_weight
        u = k / N  # normalized progress in [0,1)
        z = self._a * (u - 0.5)
        # numerically stable sigmoid
        if z >= 0:
            return 1.0 / (1.0 + math.exp(-z)) * self.max_weight
        ez = math.exp(z)
        return ez / (1.0 + ez) * self.max_weight

if __name__ == "__main__":
    import plotly.graph_objects as go
    
    scheduler = SigmoidFadeIn(max_weight=0.01, start_step=10000, duration=30000, step_size=10)
    
    steps = list(range(0, 40_000, 1))
    weights = [scheduler(step) for step in steps]
    
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=weights, mode='lines', name='Weight Schedule'))
    fig.update_layout(title='Sigmoid Fade-In Schedule', xaxis_title='Step', yaxis_title='Weight')
    fig.show()