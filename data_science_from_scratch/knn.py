from collections import Counter

def knn_classify(k, labeled_points, new_point):
    # each labeled point should be a pair (point, label)
    by_distance = sorted(labeled_points, key=lambda (point, _): distance(point, new_point))
    # find the label for k closest
    k_nearest_labels = [label for _, label in by_distance[:k]]
    return majority_vote(k_nearest_labels)


def majority_vote(labels):
    # assume that labels are for points ordered with nearest to farthest labeled_points
    vote_counts = Counter(labels)
    winner, winner_count = vote_counts.most_common(1)[0]
    num_winners = len([count for count in vote_counts.values() if count==winner_count])

    if num_winners == 1:
        # return winner when no tie
        return winner
    # when there is a tie, try again without the farthest
    return majority_vote(labels[:-1])


# ANN is much faster implementation of KNN
