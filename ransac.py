def traditional(num_inliers, num_samples, sample_size):
    return 1 - (num_inliers / num_samples)**sample_size

def correct(num_inliers, num_samples, sample_size):
    prob = 1
    for i in range(sample_size):
        prob *= (num_inliers - i) / (num_samples - i)
    return 1 - prob

print(traditional(19, 20, 5))
print(correct(19, 20, 5))
