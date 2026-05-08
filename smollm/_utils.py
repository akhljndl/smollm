import math


def cosine_lr(step: int, warmup: int, total: int, max_lr: float) -> float:
    if step < warmup:
        return max_lr * step / warmup
    progress = (step - warmup) / max(1, total - warmup)
    return max_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
