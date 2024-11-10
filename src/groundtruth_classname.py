
# Content of src/groundtruth_classname.py
import yaml
import logging

def load_class_names(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data['names']

def parse_label(label_data, class_names):
    parts = label_data.strip().split()
    class_id = int(parts[0])
    class_name = class_names[class_id]
    coordinates = [float(x) for x in parts[1:]]
    return class_name, coordinates

def verify_detection(detected_class, ground_truth_label, class_names):
    try:
        if isinstance(ground_truth_label, tuple):
            gt_class, _ = ground_truth_label
        else:
            gt_class, _ = parse_label(ground_truth_label, class_names)

        if detected_class == gt_class:
            return True, f"Correct detection: {detected_class}"
        else:
            return False, f"Incorrect detection. Detected: {detected_class}, Ground Truth: {gt_class}"
    except Exception as e:
        logging.error(f"Error verifying detection: {e}")
        return False, f"Error in verification process: {e}"

def load_ground_truth(label_file_path, class_names):
    try:
        with open(label_file_path, 'r') as file:
            lines = file.readlines()
        return [parse_label(line, class_names) for line in lines]
    except Exception as e:
        logging.error(f"Error loading ground truth data: {e}")
        return []
