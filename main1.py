import cv2
import torch
import numpy as np
from ultralytics import YOLO
import easyocr
from scipy.special import softmax
import os
import logging
from fuzzywuzzy import fuzz
from transformers import AutoTokenizer, AutoModelForMaskedLM
from difflib import get_close_matches

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Product database
PRODUCT_DATABASE = [
    "BargsBlack20Oz", "BirdDogApple750Ml", "BirdDogBlackCherry750Ml", "BirdDogSaltedCaramelWhiskey750Ml",
    "BirdDogStrawberry750Ml", "BuenoShareSize", "CheetosCrunchy", "CheetosCrunchyFlaminHot",
    "CheetosCrunchyFlaminHotLimon", "CheetosCrunchyXXTRAFlaminHot", "CheetosCrunchyXXTRAFlaminHotLimon",
    "CheetosFlaminHotPuffs", "CheetosPuffs", "CheetosPuffsWhiteCheddar", "CherryCocaCola20Oz",
    "CherryVanillaCocaCola20Oz", "ChestersFriesFlaminHot", "ChipsAhoy", "ChipsAhoyKingSize",
    "CocaCola16Oz", "CocaCola20Oz", "CocaCola350Ml", "CocaColaSpiced20Oz", "CocaColaZero16Oz",
    "Crunch", "Crush16Oz", "DekuyperButtershots200Ml", "DekuyperButtershots750Ml", "DekuyperHotDamn100Pr200Ml",
    "DekuyperHotDamn30Pr200Ml", "DekuyperHotDamn30Pr750Ml", "DekuyperPeachtree200Ml", "DekuyperPeachtree750Ml",
    "DekuyperPeppermint100Pr750Ml", "DietCocaCola20Oz", "DietCokeCan16Oz", "DoritosCoolRanch",
    "DoritosDinamitaChileLimon", "DoritosDinamitaFlaminHotQueso", "DoritosDinamitaSticksHotHoneyMustard",
    "DoritosDinamitaSticksSmokyChileQueso", "DoritosDinamitaSticksTangyFieryLime", "DoritosFlaminHotCoolRanch",
    "DoritosNachoCheese", "DoritosSpicyNacho", "DrPapPepper1L", "DrPapPepper20Oz", "DrPapPepperCan16Oz",
    "EarlyTimes200Ml", "Fanta20Oz", "FantaCan16Oz", "FantaGrape20Oz", "FantaZero20Oz",
    "FritoLayBBQSunflowerSeeds", "FritoLayRanchSunflowerSeeds", "FritosChiliCheese", "FunyunsOnionFlavoredRings",
    "FunyunsOnionFlavoredRingsFlaminHot", "GrandmasChocolateBrownieCookies", "GrandmasChocolateChipCookies",
    "GrandmasMiniChocolateChipCookies", "GrandmasMiniSandwichCremesVanillaFlavoredCookies",
    "GrandmasOatmealRaisinCookies", "GrandmasPeanutButterCookies", "GrandmasSandwichCremePeanutButterCookies",
    "GrandmasSandwichCremeVanillaFlavoredCookies", "JimBeamApple750Ml", "JimBeamBurbon200Ml", "JimBeamFire750Ml",
    "JimBeamFire750MlPet", "JimBeamHoney200Ml", "JimBeamHoney750MlPet", "JimBeamRedStag200Ml", "JimBeamRedStag750Ml",
    "LaysBarbecue", "LaysClassic", "LaysKettleCookedJalapeno", "LaysLimon", "LennyLarrysBirthdayCake",
    "LennyLarrysChocolateChips", "LennyLarrysDoubleChocolateChips", "LennyLarrysPeanutButter",
    "LennyLarrysPeanutButterChocolateChips", "LennyLarrysSnickerdoodle", "MinuteMaidBlueRaspberry20Oz",
    "MinuteMaidFruitPunch", "MinuteMaidLemonade20Oz", "MinuteMaidPinkLemonade", "MountainDew16Oz", "MtnDew16Oz",
    "MunchiesFlamingHotPeanuts", "MunchiesHoneyRoastedPeanuts", "MunchiesPeanutButter", "MunchosPotatoCrisps",
    "NerdsShareSize", "NutHarvestTrailMixCocoaDustedWholeAlmonds", "NutHarvestTrailMixDeluxeSaltedMixedNuts",
    "NutHarvestTrailMixHoneyRoastedWholeCashews", "NutHarvestTrailMixLightlyRoastedWholeAlmonds",
    "NutHarvestTrailMixNutChocolateSweetSalty", "NutHarvestTrailMixNutFruitSweetSalty",
    "NutHarvestTrailMixSeaSaltedInShellPistachios", "NutHarvestTrailMixSeaSaltedWholeCashews",
    "NutHarvestTrailMixSpicyFlavoredPistachios", "Oreo", "OreoDoubleStufKingSize", "OreoKingSize",
    "PaydayShareSize", "PopcornersKettleCorn", "PopcornersSeaSalt", "PopcornersWhiteCheddar",
    "QuakerCaramelRiceCrisps", "RufflesCheddarSourCream", "RufflesQueso", "SabritasJapanese", "SabritasSaltLime",
    "SabritasSpicyPicante", "SabritonesChileLime", "SkittlesShareSize", "SmartfoodWhiteCheddar",
    "SourPunchShareSize", "Sprite20Oz", "Sprite40Oz", "SpriteCan16Oz", "SpriteCherry20Oz", "SpriteTropicalMix20Oz",
    "SpriteZero20Oz", "SunChipsHarvestCheddar", "VanillaCocaCola20Oz", "WhatchamacallitKingSize", "ZeroCocaCola16Oz"
]

def check_file_exists(file_path):
    if not os.path.isfile(file_path):
        logging.error(f"File not found: {file_path}")
        return False
    return True

class ObjectDetector:
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.1):
        try:
            self.model = YOLO(model_path)
            logging.info(f"YOLO model loaded: {self.model}")
        except Exception as e:
            logging.error(f"Failed to load YOLO model: {e}")
            raise
        self.conf_threshold = conf_threshold
        self.relevant_classes = ['bottle', 'cup', 'can', 'bowl', 'box', 'carton', 'package', 'book']

    def detect(self, frame):
        try:
            results = self.model(frame, conf=self.conf_threshold)
            all_detections = []
            relevant_detections = []
            for r in results:
                for box in r.boxes:
                    cls = int(box.cls)
                    conf = float(box.conf)
                    class_name = self.model.names[cls]
                    all_detections.append((class_name, conf))
                    if class_name in self.relevant_classes:
                        relevant_detections.append((box.xyxy[0].tolist(), conf, class_name))
            logging.info(f"All detected objects: {all_detections}")
            logging.info(f"Relevant detections: {relevant_detections}")
            return relevant_detections
        except Exception as e:
            logging.error(f"Error in object detection: {e}")
            return []

class OCRProcessor:
    def __init__(self, lang=['en']):
        try:
            self.reader = easyocr.Reader(lang)
            self.tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
            self.model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")
            logging.info("OCR and BERT models loaded successfully")
        except Exception as e:
            logging.error(f"Failed to initialize OCR or BERT: {e}")
            raise

    def perform_ocr(self, img):
        try:
            results = self.reader.readtext(img)
            filtered_results = [r for r in results if r[2] > 0.3]
            return filtered_results
        except Exception as e:
            logging.error(f"OCR failed: {e}")
            return []

    def extract_product_name(self, ocr_results):
        try:
            extracted_text = " ".join([text for _, text, _ in ocr_results])
            words = extracted_text.split()

            for i, word in enumerate(words):
                if '[UNK]' in word or len(word) < 3:
                    masked_text = " ".join(words[:i] + ['[MASK]'] + words[i+1:])
                    predictions = self.bert_predict(masked_text)
                    best_prediction = get_close_matches(word, predictions, n=1, cutoff=0.6)
                    if best_prediction:
                        words[i] = best_prediction[0]

            corrected_text = " ".join(words)
            best_match = None
            best_ratio = 0
            for product in PRODUCT_DATABASE:
                ratio = fuzz.partial_ratio(corrected_text.lower(), product.lower())
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_match = product
            return best_match if best_ratio > 70 else corrected_text, best_ratio / 100
        except Exception as e:
            logging.error(f"Product name extraction failed: {e}")
            return "", 0

    def bert_predict(self, text, max_predictions=5):
        try:
            inputs = self.tokenizer(text, return_tensors="pt")
            mask_token_index = torch.where(inputs["input_ids"] == self.tokenizer.mask_token_id)[1]

            with torch.no_grad():
                outputs = self.model(**inputs)

            logits = outputs.logits
            mask_token_logits = logits[0, mask_token_index, :]
            top_tokens = torch.topk(mask_token_logits, max_predictions, dim=1).indices[0].tolist()

            return [self.tokenizer.decode([token]) for token in top_tokens]
        except Exception as e:
            logging.error(f"BERT prediction failed: {e}")
            return []

class ProductDetectionSystem:
    def __init__(self, model_path='yolov8n.pt', conf_threshold=0.1, ocr_threshold=0.3):
        self.detector = ObjectDetector(model_path, conf_threshold)
        self.ocr_processor = OCRProcessor()
        self.ocr_threshold = ocr_threshold

    def process_frame(self, frame):
        logging.info("Processing frame")
        try:
            detections = self.detector.detect(frame)
            logging.info(f"Detected objects: {detections}")

            results = []
            if not detections:
                logging.info("No relevant objects detected, performing OCR on entire frame")
                ocr_result = self.ocr_processor.perform_ocr(frame)
                logging.info(f"OCR Result for entire frame: {ocr_result}")

                if ocr_result:
                    product_name, ocr_conf = self.ocr_processor.extract_product_name(ocr_result)
                    results.append({
                        'bbox': (0, 0, frame.shape[1], frame.shape[0]),
                        'class': 'full_frame',
                        'conf': 1.0,
                        'ocr_text': product_name,
                        'ocr_conf': ocr_conf
                    })
            else:
                for bbox, conf, class_name in detections:
                    x1, y1, x2, y2 = map(int, bbox)
                    roi = frame[y1:y2, x1:x2]
                    ocr_result = self.ocr_processor.perform_ocr(roi)
                    logging.info(f"OCR Result for {class_name}: {ocr_result}")

                    product_name, ocr_conf = self.ocr_processor.extract_product_name(ocr_result)

                    results.append({
                        'bbox': (x1, y1, x2, y2),
                        'class': class_name,
                        'conf': conf,
                        'ocr_text': product_name,
                        'ocr_conf': ocr_conf
                    })

            # Draw bounding boxes and labels
            for result in results:
                x1, y1, x2, y2 = result['bbox']
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                label = f"{result['class']}: {result['conf']:.2f}"
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                ocr_label = f"OCR: {result['ocr_text']} ({result['ocr_conf']:.2f})"
                cv2.putText(frame, ocr_label, (x1, y2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

            return frame, results
        except Exception as e:
            logging.error(f"Frame processing failed: {str(e)}")
            return frame, []

    def process_static_image(self, image_path):
        logging.info(f"Processing image: {image_path}")
        if not check_file_exists(image_path):
            return

        try:
            frame = cv2.imread(image_path)
            if frame is None:
                raise FileNotFoundError(f"Could not read the image: {image_path}")

            logging.info(f"Image shape: {frame.shape}")
            processed_frame, results = self.process_frame(frame)

            if not results:
                logging.info("No products detected in the image.")
            else:
                for result in results:
                    logging.info(f"Detected: {result['class']} (Conf: {result['conf']:.2f}), "
                                 f"OCR: {result['ocr_text']} (Conf: {result['ocr_conf']:.2f}), "
                                 f"BBox: {result['bbox']}")

            cv2.imshow('Processed Image', processed_frame)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            logging.error(f"Error processing static image: {e}")

    def run_webcam(self):
        logging.info("Starting webcam")
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise IOError("Cannot open webcam")

            while True:
                ret, frame = cap.read()
                if not ret:
                    logging.error("Failed to grab frame")
                    break

                processed_frame, results = self.process_frame(frame)
                cv2.imshow('Product Detection', processed_frame)

                for result in results:
                    logging.info(f"Detected: {result['class']} (Conf: {result['conf']:.2f}), "
                                 f"OCR: {result['ocr_text']} (Conf: {result['ocr_conf']:.2f}), "
                                 f"BBox: {result['bbox']}")

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            cap.release()
            cv2.destroyAllWindows()
        except Exception as e:
            logging.error(f"Webcam processing failed: {e}")

def main():
    logging.info("Starting Product Detection System")
    try:
        system = ProductDetectionSystem(conf_threshold=0.1, ocr_threshold=0.3)

        current_dir = os.path.dirname(os.path.abspath(__file__))
        test_images = [
            os.path.join(current_dir, 'test1.jpg'),
            os.path.join(current_dir, 'test2.jpg')
        ]

        for img in test_images:
            system.process_static_image(img)

        use_webcam = input("Do you want to try the webcam? (y/n): ").lower().strip() == 'y'
        if use_webcam:
            system.run_webcam()
        else:
            logging.info("Webcam test skipped. Exiting program.")
    except Exception as e:
        logging.error(f"An error occurred in the main execution: {e}")

if __name__ == "__main__":
    main()