from fractions import Fraction
from exact_reward import payouts


def strict_rewards(scores):
    ranks = [[Fraction(n, d) for n, d in row] for row in scores["rr"]]
    mean = [sum(row) / len(row) for row in ranks]
    details = scores["detail"]
    anchor = ranks[-1]
    joint = []
    for i, row in enumerate(ranks[:-1]):
        improved = sum(value > base for value, base in zip(row, anchor))
        passes = details[i] > details[-1] and mean[i] > mean[-1]
        joint.append(Fraction(improved, len(row)) if passes else Fraction(0))
    return {"ordinary": payouts(mean[:-1]), "joint": payouts(list(zip(joint, mean[:-1])))}, [float(x) for x in joint]
