def test_product_detection_system():
    system = ProductDetectionSystem(conf_threshold=0.1)  # Lower threshold for testing
    test_image = cv2.imread('test1.jpg')
    if test_image is None:
        raise FileNotFoundError("Test image 'test1.jpg' not found")
    print(f"Test image shape: {test_image.shape}")
    processed_frame, fused_results = system.process_frame(test_image)
    assert processed_frame is not None, "Frame processing failed"
    print("Product Detection System test completed")
    return len(fused_results) > 0

if __name__ == "__main__":
    try:
        test_passed = test_product_detection_system()
        print(f"Product Detection System Test: {'Passed' if test_passed else 'Failed'}")
    except Exception as e:
        print(f"An error occurred during testing: {str(e)}")

    system = ProductDetectionSystem(conf_threshold=0.1)  # Lower threshold for main execution

    for img in ['test1.jpg', 'test2.jpg']:
        try:
            system.process_static_image(img)
        except FileNotFoundError as e:
            print(e)

    use_webcam = input("Do you want to try the webcam? (y/n): ").lower().strip() == 'y'

    if use_webcam:
        system.run_webcam()
    else:
        print("Webcam test skipped. Exiting program.")