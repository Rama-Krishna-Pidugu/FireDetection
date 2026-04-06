import numpy as np
with open('out2.txt', 'w', encoding='utf-8') as f:
    def dist(exp):
        scores = []
        for t in range(5):
            for r in range(5):
                for w in range(5):
                    lin = (t + w + (4 - r)) / 12.0
                    scores.append(np.round((lin**exp)*4).astype(int))
        unique, counts = np.unique(scores, return_counts=True)
        f.write(f"Exp {exp}: {dict(zip(unique, counts))}\n")

    dist(1.0)
    dist(1.5)
    dist(2.0)
    dist(2.5)
    dist(3.0)
