import math

class ExpWeightAnnealer:
    """1 -> 0 stepwise exponential decay starting at start_step."""
    def __init__(self, start_step=5_000, base=0.99, step_size=5):
        assert 0.0 < base < 1.0
        self.start_step = start_step
        self.base = base
        self.step_size = max(1, step_size)

    def __call__(self, step: int) -> float:
        if step < self.start_step:
            return 1.0
        k = (step - self.start_step) // self.step_size  # integer count of decay steps
        # Use exp/log for stability when k is large:
        w = math.exp(k * math.log(self.base))
        return w

def approx(a, b, tol=1e-9):
    return abs(a - b) <= tol * max(1.0, abs(a), abs(b))

def run_basic_tests():
    # Case 1: before start_step → 1.0
    A = ExpWeightAnnealer(start_step=10, base=0.9, step_size=5, min_w=1e-6)
    assert A(0) == 1.0
    assert A(9) == 1.0

    # Case 2: at start_step → still 1.0 (k = 0)
    assert A(10) == 1.0

    # Case 3: first decay step after step_size
    # step=15 → k=(15-10)//5 = 1 → base**1 = 0.9
    assert approx(A(15), 0.9)

    # Case 4: multiple steps
    # step=25 → k=3 → 0.9**3
    expected = 0.9 ** 3
    assert approx(A(25), expected)

    # Case 5: min_w cutoff to zero
    B = ExpWeightAnnealer(start_step=0, base=0.1, step_size=1, min_w=1e-3)
    # 0.1**2 = 0.01 (>1e-3), 0.1**4 = 1e-4 (<1e-3) → should clamp to 0.0
    assert approx(B(2), 0.01)
    assert B(4) == 0.0

    # Case 6: step_size=1 behaves like every-step decay
    C = ExpWeightAnnealer(start_step=3, base=0.5, step_size=1, min_w=0.0)
    # steps: 0..2 => 1.0; step=3 => 1.0; step=4 => 0.5; step=5 => 0.25
    assert C(2) == 1.0
    assert C(3) == 1.0
    assert approx(C(4), 0.5)
    assert approx(C(5), 0.25)

    print("All basic tests passed!")

def demo_print():
    A = ExpWeightAnnealer(start_step=10, base=0.99, step_size=5, min_w=1e-4)
    print("\nDemo: stepwise weights")
    print("step\tweight")
    for s in range(0, 401, 5):
        print(f"{s}\t{A(s):.8f}")


if __name__ == "__main__":
    run_basic_tests()
    demo_print()