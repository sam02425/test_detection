
# Content of src/evaluation_metrics.py

def calculate_metrics(results):
    true_positives = 0
    false_positives = 0
    false_negatives = 0

    for result in results:
        detections = result['detections']
        ground_truth = result['ground_truth']

        matched_detections = set()
        matched_ground_truth = set()

        for verified_detection in detections:
            detection = verified_detection['detection']
            is_correct = verified_detection['is_correct']

            if is_correct and detection['class'] not in matched_ground_truth:
                true_positives += 1
                matched_ground_truth.add(detection['class'])
                matched_detections.add(detection)
            else:
                false_positives += 1

        false_negatives += len(ground_truth) - len(matched_ground_truth)

    precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
    recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return {
        'precision': precision,
        'recall': recall,
        'f1_score': f1_score,
        'true_positives': true_positives,
        'false_positives': false_positives,
        'false_negatives': false_negatives
    }
