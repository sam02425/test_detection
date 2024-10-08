import cv2
import logging
import numpy as np
from ultralytics import YOLO
from src.ocr_processor import OCRProcessor
from src.product_matcher import ProductMatcher, ImprovedProductMatcher
from collections import Counter
from difflib import SequenceMatcher
import heapq
from src.product_detection_system import ProductDetectionSystem

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

PRODUCT_DATABASE = ['1792 BOTTLED IN BOND 750ML', '1792 FULL PROOF 750ML', '1792 SMALL BATCH 750ML', '1792 SWEET WHEAT 750ML', '1800-COCONUT-TEQUILA-750ML', '1800-CRISTALINO-ANEJO-750ML', '1800-SILVER-BLANCO-TEQUILA-750ML', '360 BLUE RASPBERRY 50ML', '360 PINEAPPLE 50ML', '99 APPLE 100ML', '99 APPLE 50 ML', '99 BANANAS 100ML', '99 BANANAS 50ML', '99 BLUE RASPBERRY 50 ML', '99 BUTTERSCOTCH 100ML', '99 CHERRY LIMEADE 50ML', '99 CHOCOLATE 50ML', '99 FRUIT PUNCH 50ML', '99 GRAPES 100ML', '99 MYSTERY FLAVOR 50ML', '99 PEACH 50ML', '99 PEACHES 100ML', '99 PEPPERMINT 50ML', '99 ROOT BEER 50 ML', '99 SOUR APPLE 50ML', '99 SOUR BERRY 50ML', '99 SOUR CHERRY 50ML', '99 WHIPPED 50ML', 'ABSOLUT 80 375ML', 'ADMIRAL-NELSON-S-SPICED-RUM-1.75L', 'ADMIRAL-NELSON-S-SPICED-RUM-750ML', 'AVION-SILVER-TEQUILA-750ML', 'BACARDI GOLD 375ML', 'BACARDI SUPERIOR 375ML', 'BACARDI SUPERIOR 50ML', 'BAILEYS IRISH CREAM 375ML', 'BAILEYS IRISH CREAM 50ML', 'BAKERS BBN 13YR 750ML', 'BANKS-ISLAND-BLEND-5-RUM-750ML', 'BASIL HAYDEN 10YR 750ML', 'BASIL HAYDEN 375ML', 'BASIL HAYDEN 750ML', 'BELVEDERE VODKA 375ML', 'BENCHMARK 50ML', 'BENJAMINS PINEAPPLE 50ML', 'BENJAMINS WATERMELON 50ML', 'BIRD DOG 7YR 50ML', 'BIRD DOG APPLE WHISKEY 50 ML', 'BIRD DOG BLACK CHERRY 50ML', 'BIRD DOG BLACKBERRY WHISKEY 50 ML', 'BIRD DOG CANDY CANE WHISKEY 50 ML', 'BIRD DOG CHOCOLATE WHISKEY 50 ML', 'BIRD DOG GINGER BREAD 50 ML', 'BIRD DOG MESQUITE BROWN SUGAR 50ML', 'BIRD DOG PEANUT BUTTER 50ML', 'BIRD DOG PUMPKIN SPICE WHISKEY 50 ML', 'BIRD DOG SALTED CARAMEL 50 ML', 'BIRD DOG SMORES WHISKEY 50ML', 'BIRD DOG STRAWBERRY 50ML', 'BLOOD OATH PACT NO 9 750ML', 'BLOOD OATH PACT NO.7 750ML', 'BOMBAY BRAMBLE GIN 50ML', 'BOMBERGER-S DECLARATION BOURBON 750ML', 'BOOKERS LITTLE BOOK 122.6PR', 'BOOKERS NOE 2021-02 750ML', 'BOZAL-ENSEMBLE-MEZCAL-750ML', 'BRINLEY-GOLD-SHIPWRECK-COCONUT-RUM-750ML', 'BRINLEY-GOLD-SHIPWRECK-COFFEE-RUM-750ML', 'BRINLEY-GOLD-SHIPWRECK-SPICED-RUM-750ML', 'BRINLEY-GOLD-SHIPWRECK-VANILLA-RUM-750ML', 'BRUGAL-1888-DOBLEMENTE-ANEJADO-750ML', 'BUFFALO TRACE BOURBON 750ML', 'BUFFALO TRACE BOURBON CREAM 375ML', 'BULLEIT BLENDERS SELECT 750ML', 'BULLEIT BOURBON 90PR 375ML', 'BUMBU-CREME-RUM-750ML', 'BUMBU-CREME-RUM-BOXED-750ML', 'BUMBU-RUM-750ML', 'CALUMET FARM 12YR BOURBON 750ML', 'CAMARENA-ANEJO-TEQUILA-750ML', 'CANADIAN CLUB 375ML', 'CANADIAN MIST 375ML', 'CANADIAN RESERVE 375 ML', 'CAPTAIN MORGAN SPICED RUM 375ML', 'CASA-DRAGONES-ANEJO-TEQUILA-750ML', 'CASA-DRAGONES-BLANCO-TEQUILA-750ML', 'CASA-DRAGONES-JOVEN-SIPPING-TEQUILA-750ML', 'CASA-NOBLE-TEQUILA-BLANCO-750ML', 'CASAMIGOS BLANCO TEQUILA 750ML', 'CASAMIGOS REPOSADO TEQUILA 750ML', 'CASTILLO-SILVER-RUM-PT-750ML', 'CASTLE - KEY SMALL BATCH BOURBON 750ML', 'CHARTREUSE 110 GREEN 750ML', 'CHARTREUSE 80 YELLOW 750ML', 'CHRISTIAN BROTHERS VS 375ML', 'CLASE-AZUL-REPOSADO-TEQUILA-750ML', 'COA-REPOSADO-TEQUILA-750ML', 'COA-SILVER-TEQUILA-750ML', 'CODIGO-1530-REPOSADO-TEQUILA-750ML', 'CODIGO-1530-ROSA-BLANCO-TEQUILA-750ML', 'COOPERS CRAFT 50 ML', 'COPPER TONGUE 16YR 750ML', 'CORAZON-BLANCO-TEQUILA-750ML', 'CORRALEJO BLANCO TEQUILA 750ML', 'CREAM OF KENTUCKY 13', 'CROWN ROYAL 18YRS EXTRA RARE 750ML', 'CROWN ROYAL 375ML', 'CROWN ROYAL 50 ML', 'CROWN ROYAL APPLE 375ML', 'CROWN ROYAL APPLE 50 ML', 'CROWN ROYAL EXTRA RARE 750ML', 'CROWN ROYAL PEACH 375ML', 'CROWN ROYAL PEACH 50ML', 'CROWN ROYAL VANILLA 200ML', 'CROWN ROYAL VANILLA 375ML', 'CRUZAN-9-SPICED-RUM-750ML', 'CRUZAN-BLACK-CHERRY-RUM-750ML', 'CRUZAN-BLACK-STRAP-RUM-750ML', 'Captain Morgan Private Stock Rum- 750ml', 'DANCING-PINES-SPICE-RUM-750ML', 'DARK EYES 100PR 375ML', 'DARK EYES 80PR 375ML', 'DARK EYES 80PR 50ML', 'DARK EYES CHERRY 375ML', 'DEKUYPER APPLE PUCKER 375ML', 'DEKUYPER BUTTERSHOTS 375ML', 'DEKUYPER HOT DAMN 100 375ML', 'DEKUYPER HOT DAMN 30 PR 375ML', 'DEKUYPER PEACHTREE 375ML', 'DEKUYPER PEPPERMINT 100PR 375ML', 'DEKUYPER PEPPERMINT 60 50ML', 'DEKUYPER PEPPERMINT 60PR 375ML', 'DELEOB-BLANCO-TEQUILA-750ML', 'DEWARS WHITE LABEL 375ML', 'DISARONNO 375ML', 'DOM B-B LIQUEUR 375ML', 'DOM BENEDICTINE 375ML', 'DOMAINE CANTON 375ML', 'DON JULIO BLANCO 375ML', 'DON JULIO REPOSADO 375ML', 'DON-JULIO-1942-TEQUILA-750ML', 'DON-JULIO-70-1942-750ML', 'DON-JULIO-70-1942-BOXED-750ML', 'DON-JULIO-ANEJO-750ML', 'DON-JULIO-BLANCO-1.75ML', 'DON-JULIO-BLANCO-750ML', 'DONJULIO-PRIMAVERA-REPOSADO-TEQUILA-750ML', 'DR MCGILLICUDDY MENTHOLMINT 375 ML', 'DR MCGILLICUDDY MENTHOLMINT SCHNAPPS 50ML', 'DRAMBUIE -SCOT- 375ml', 'DUKE FOUNDERS RESERVE 110PR', 'DULCE-VIDA-EXTRA-ANEJO-TEQUILA-750ML', 'DULCE-VIDA-LIME-PL', 'DULCE-VIDA-ORGANIC-ANEJO-100PR-750ML', 'DULCE-VIDA-ORGANIC-BLANCO-100PR-TEQUILA-750ML', 'DULCE-VIDA-PINEAPPLE-JALAPENO-TEQUILA-750ML', 'DUSSE VSOP COGNAC 375ML', 'Del Maguey VIDA de San Luis del Rio 750ml', 'E-J BRANDY VS 375ML', 'E.H. TAYLOR SINGLE BARREL 750ML', 'E.H. TAYLOR STRAIGHT RYE 750ML', 'EAGLE RARE 10YR SINGLE BARREL 750ML', 'EARLY TIMES 375ML', 'EARLY TIMES 50ML', 'EARLY TIMES BOTTLED IN BOND 100PR 1LT', 'EL JIMADOR REPOSADO 375ML', 'EL JIMADOR SILVER 375ML', 'EL-JIMADOR-ANEJO-TEQUILA-750ML', 'EL-JIMADOR-REPOSADO-TEQUILA-750ML', 'EL-JIMADOR-SILVER-TEQUILA-750ML', 'ELIJAH CRAIG BARREL PROOF 750ML', 'ELIJAH CRAIG SB 18YR 90 PROOF', 'ESPOLON-TEQUILA-BLANCO-750ML', 'ESPOLON-TEQUILA-REPOSADO-750ML', 'EVAN WILLIAMS 375ML', 'EVAN WILLIAMS CHERRY 50 ML', 'EVERCLEAR 375ML', 'EXOTICO-BLANCO-TEQUILA-750ML', 'EXOTICO-REPOSADO-TEQUILA-750ML', 'EZRA BROOKS BOURBON CREAM PL', 'FIREBALL CINNAMON 100ML', 'FIREBALL CINNAMON 375ML', 'FIREBALL CINNAMON 50ML', 'GEORGE DICKEL BOTTLED IN BOND WISHKEY 13YR', 'GEORGE T STAGG 750ML', 'GOSLINGS-BLACK-SEAL-RUM-750ML', 'GRAN PATRON PLATINUM 375ML', 'GREY GOOSE 375ML', 'GRIND CARAMEL 50ML', 'HARD TRUTH CINNAMON VODKA 50ML', 'HARD-TRUTH-TOASTED-COCONUT-RUM-750ML', 'HARD-TRUTH-TOASTED-COCONUT-RUM-CREAM-750ML', 'HENNESSY VS 375ML', 'HENNESSY VS COGNAC 200ML 80PR', 'HERRADURA-SILVER-TEQUILA-750ML', 'HERRADURA-ULTRA-ANEJO-TEQUILA-750ML', 'HERRADURA-ULTRA-ANEJO-TEQUILA-BOXED-750ML', 'HIGH WEST A MIDWINTER NIGHTS DRAM 750ML', 'HIGH WEST RENDEZVOUS 750ML', 'HORNITOS-PLATA-TEQUILA-750ML', 'HORNITOS-REPOSADO-TEQUILA-750ML', 'HORSE SOLDIER BB', 'JACK DANIEL-S 10YR 700ML', 'JACK DANIELS APPLE 375ML', 'JACK DANIELS APPLE 50ML', 'JACK DANIELS BLACK 375ML', 'JACK DANIELS BLACK 50 ML', 'JACK DANIELS FIRE 375ML', 'JACK DANIELS FIRE 50 ML', 'JACK DANIELS HONEY 375ML', 'JACK DANIELS SINGLE BARREL 375ML', 'JACK DANIELS SINGLE BARREL 50ML', 'JACK DANIELS SINGLE BARREL ERIC CHURCH 750ML', 'JAGERMEISTER 375ML', 'JAGERMEISTER 50ML', 'JEFFERSONS OCEAN AGE AT SEA 375ML', 'JIM BEAM 375ML', 'JIM BEAM 50 ML', 'JIM BEAM 8 STAR 375ML', 'JIM BEAM APPLE 375ML', 'JIM BEAM APPLE 50ML', 'JIM BEAM BLACK 375ML', 'JIM BEAM DISTILLER-S CUT 750ML', 'JIM BEAM FIRE 375', 'JIM BEAM HONEY 375ML', 'JIM BEAM HONEY 50 ML', 'JIM BEAM PEACH 375', 'JIM BEAM PEACH 50ML', 'JIM BEAM RED STAG 375ml', 'JIM BEAM RED STAG 50ML', 'JIM BEAM VANILLA 375ML', 'JOHNNIE WALKER BLACK 375ML', 'JOHNNIE WALKER BLACK LABEL 50ML', 'JOHNNIE WALKER BLUE 750ML REG', 'JOHNNIE WALKER RED LABEL 200ML', 'JOHNNIE-WALKER-BLUE-YEAR-OF-PIG-SCOTCH-WHISKY-750ML', 'JOSE CUERVO GOLD 375 ML', 'JOSE CUERVO GOLD 50ML', 'JOSE CUERVO SILVER 375ML', 'JOSE-CUERVO-GOLD-TEQUILA-1.75L', 'JOSE-CUERVO-GOLD-TEQUILA-750ML', 'JOSE-CUERVO-SILVER-TEQUILA-750ML', 'JOSEPH MAGNUS CIGAR BLEND BBN', 'KETEL ONE 375ML', 'KINKY LIQUEUR GREEN 50 ML', 'KIRK-AND-SWEENEY-RESERVA-RUM-750ML', 'KNOB CREEK BOURBON 375ML', 'KOMOS-ANEJO-CRISTAL-TEQUILA-750-ML', 'KOMOS-ANEJO-CRISTAL-TEQUILA-BOXED-750-ML', 'LARCENY BARREL PROOF 750ML', 'LEBLON-CACHACA-BRASIL-750ML', 'MAESTRO-DOBEL-DIAMANTE-TEQUILA-750ML', 'MAESTRO-DOBEL-DIAMANTE-TEQUILA-BOXED-750ML', 'MAKER-S MARK WOOD FINISH SERIES BOURBON 2023 750ML', 'MAKERS MARK BOURBON 375ML', 'MAKERS MARK BRT-O1 LTD 2022', 'MALIBU 375ML', 'MCCORMICKS BLEND 375ML', 'MEZCAL-NUCANO-ANEJO-750ml-80proof', 'MEZCAL-NUCANO-JOVEN-750ml-90proof', 'MEZCAL-NUCANO-REPOSADO-750ml-80proof', 'MEZCALES-MALA-IDEA-TOBALA-750ML', 'MICHTER-S SOUR MASH BOURBON 750ML', 'MOUNT-GAY-BLACK-BARREL-750ML', 'MYERS-S-ORIGINAL-DARK-RUM-750ML', 'Malibu Caribbean Rum with Coconut Flavored Liqueur 750mL 42 Proof', 'Malibu-Black-Caribbean-Rum-with-Coconut-Flavored-Liqueur-750mL-70-Proof', 'Malibu-Caribbean-Rum-with-Coconut-Flavored-Liqueur-1.75L-42-Proof', 'NATURAL LIGHT LEMONADE VODKA 50ML', 'NEW AMSTERDAM PINK WHITNEY 375 ML', 'NOAHS MILL SMALL BATCH BOURBON 750ML', 'NUMBER-JUAN-BLANCO-TEQUILA-750ML', 'NUMBER-JUAN-REPOSADO-TEQUILA-750ML', 'OLD CAMP PEACH PECAN 100ML', 'OLD EZRA 7 YR BARREL STRENGTH 750ML', 'OLD FITZGERALD 19YR 750ML', 'OLD FITZGERALD BOTTLED IN BOND 8YR 750ML', 'OLD FORESTER 1910 750ML', 'OLE SMOKY APPLE PIE 50ML', 'OLE SMOKY BANANA PUDDING 50ML', 'OLE SMOKY BLACKBERRY 50ML', 'OLE SMOKY MOUNTAIN JAVA 50ML', 'OLE SMOKY SALTY CARAMEL 375ML', 'OLE SMOKY SALTY CARAMEL WHISKEY 50ML', 'OLE SMOKY SOUR WATERMELON 50ML', 'OTR AVIATION 200ML', 'OTR COSMOPOLITAN 375ML', 'OTR JALAPINO PINEAPPLE MARGARITA 375ML', 'OTR OLD FASHIONED 200ML', 'OTR THE AVIATION 375ML', 'OTR THE MARGARITA 375ML', 'PAPA-S-PILAR-DARK-RUM-SHERRY-750ML', 'PARKER-S HERITAGE-24 YR', 'PARKERS HERITAGE COLLECTION 15TH EDITION 750ML', 'PARKERS HERITAGE COLLECTION DOUBLE BARRELED BLEND BOURBON750ML', 'PATRON REPOSADO 375ML', 'PATRON SILVER TEQUILA 375ML', 'PATRON SILVER TEQUILA 50 ML', 'PATRON-REPOSADO-boxed-750ML', 'PATRON-SILVER-750ML', 'PLANTATION-AGED-5-YEAR-RUM-750ML', 'PLANTATION-PINEAPPLE-RUM-750ML', 'PUSSER-S-RUM-750ML', 'PYRAT-XO-RESERVE-RUM-750ML', 'Papa-s-Pilar-Bourbon-Barrel-Finish-Rum-750ml-ABV-43-', 'Patron Extra Anejo Tequila- 750 mL Bottle- ABV 40-', 'Patron-Extra-Anejo-Tequila-BOXED-750ML-Bottle-ABV-40-', 'QUALITY-HOUSE-RUM-1.75L', 'REBEL YELL 10YR SINGLE BARREL 750ML', 'RECUERDO-MEZCAL-750ML', 'REMUS REPEAL RESERVE 100PR -5', 'REMY MARTIN VSOP 200ML', 'RIAZUL-PLATA-750ml-abv-40-', 'RON-ZACAPA-SISTEMA-NO.23-SOLERA-750ML', 'RON-ZACAPA-SISTEMA-NO.23-SOLERA-BOXED-750ML', 'RONRICO SILVER 375ML', 'RONRICO-SILVER-CARIBBEAN-RUM-PET-1.75L', 'RONRICO-SILVER-CARIBBEAN-RUM-PT-750ML', 'RUM CHATA 375ML', 'SAILOR JERRY SPICED RUM 375ML', 'SAINT CLOUD ELENA 144 PR SB', 'SAM HOUSTON 14YR BOURBON', 'SAMUEL ADAMS UTOPIAS 2021', 'SAMUEL ADAMS UTOPIAS 2021 boxed', 'SAUZA-901-SILVER-TEQUILA-750ML', 'SAUZA-HACIENDA-GOLD-TEQUILA-750ML', 'SAUZA-HACIENDA-SILVER-TEQUILA-750ML', 'SEAGRAMS 7 375ML', 'SHENKS HOMESTEAD 2020', 'SKOL VODKA 375ML', 'SKREWBALL 200ML', 'SKREWBALL 375ML', 'SLANE IRISH WHISKEY 50ML', 'SLIP KNOT RED CASK NO.9 750ML', 'SMIRNOFF 100 375ML', 'SMIRNOFF 100PR 50ML', 'SMIRNOFF 80 375ML', 'SMIRNOFF 80PR 50ML', 'SMIRNOFF BLUE RASPBERRY LEMONADE 50ML', 'SMIRNOFF KISSED CARAMEL 50ML', 'SMIRNOFF PEPPERMINT TWIST 50ML', 'SMIRNOFF RASPBERRY 50ML', 'SMIRNOFF RED-WHITE - BERRY 50ML', 'SMIRNOFF STRAWBERRY 50ML', 'SOUTHERN COMFORT 100ML', 'SOUTHERN COMFORT 70 375ML', 'STARLIGHT-SPICED-RUM-750ML', 'STOLICHNAYA 80 375ML', 'STRNAHANS BLUE PEAK', 'SUGAR-ISLAND-SPICED-750ML', 'SUGARLANDS BANANA PUDDING 50ML', 'SUGARLANDS BUTTER PECAN SIPPING CREAM 50ML', 'SUGARLANDS ELECTRIC ORANGE SIPPING CREAM 50ML', 'SUGARLANDS MARK ROGERS AMERICAN PEACH 50ML', 'SUGARLANDS PEANUT BUTTER SIPPING CREAM 50ML', 'Sailor Jerry Spiced Rum 750 ml Bottle ABV 46- 92 Proof', 'Sailor-Jerry-Spiced-Rum-1.75L-Bottle-ABV-46-92-Proof', 'TANQUERAY 375ml', 'TANTEO-JALAPENO-TEQUILA-750ML', 'TARANTULA-AZUL-TEQUILA-750ML', 'TEARS OF LLORONA EX ANEJO', 'TEREMANA-BLANCO-TEQUILA-750ML', 'TEREMANA-REPOSADO-TEQUILA-750ML', 'THE WHISTLER IRISH CREAM 50ML', 'TIN CUP 375ML', 'TITOS 50ML', 'TRES-GENERACIONES-TEQUILA-ANEJO-750ML', 'TWISTED TEA SWEET TEA WHISKEY 50ML', 'UV BLUE RASPBERRY 50ML', 'W.L. WELLER 12YR 750ML', 'WELLER FULL PROOF 750ML', 'WHEATLEY VODKA 50ML', 'WHISTLE PIG ESTATE OAK RYE 15YR 750ML', 'WILD TURKEY 101 375ml', 'WILD TURKEY AMERICAN HONEY 375ML', 'WILD TURKEY AMERICAN HONEY 50ML', 'WINDSOR APPLE 50ML', 'WOODFORD RESERVE 375ML', 'WOODFORD RESERVE 50ML', 'WOODFORD RESERVE BATCH 118.4 PROOF BOURBON 750ML', 'WOODFORD RESERVE BATCH PROOF 124.7 750ML', 'WOODFORD RESERVE DOUBLE OAKED 375ML', 'WOODFORD RESERVE MASTER-S COLLECTION 750ML', 'WOODFORD RESERVE MTR SONOMA TRIPLE FINISH 750ML', 'Weller Reserve Bourbon', 'YPIOCA-OURO-CACHACA-750ML', 'YPIOCA-PRATA-CACHACA-750ML', 'ZAYA-RUM-750ML']


class ObjectDetector:
    def __init__(self, model_path='yolo8n-l.pt', conf_threshold=0.25):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        logging.info(f"YOLO model loaded: {self.model}")

    def detect(self, frame):
        results = self.model(frame, conf=self.conf_threshold)
        detections = []
        for r in results:
            boxes = r.boxes
            for box in boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = box.conf.item()
                cls = int(box.cls)
                class_name = self.model.names[cls]
                detections.append({
                    'bbox': [x1, y1, x2, y2],  # Changed to list for consistency
                    'class': class_name,
                    'conf': conf,
                    'detection_type': 'YOLO'
                })
        logging.info(f"YOLO Detected objects: {detections}")
        return detections

def get_final_product(detections, _, threshold=0.5):
    valid_detections = [d for d in detections if isinstance(d, dict)]
    if not valid_detections:
        return "No valid detections", 0.0

    product_scores = {}

    for detection in valid_detections:
        product = detection.get('class', '')
        conf = detection.get('conf', 0.0)
        detection_type = detection.get('detection_type', '')

        # Give more weight to YOLO detections
        weight = 2.0 if detection_type == 'YOLO' else 1.0

        if product:
            if product not in product_scores:
                product_scores[product] = []
            product_scores[product].append(conf * weight)

    if not product_scores:
        return "No consistent product detected", 0.0

    logging.info(f"Aggregated product scores: {product_scores}")

    best_product = max(product_scores, key=lambda x: sum(product_scores[x]) / len(product_scores[x]))
    avg_confidence = sum(product_scores[best_product]) / len(product_scores[best_product])

    # Normalize the confidence back to 0-1 range
    avg_confidence = min(avg_confidence / 2.0, 1.0)  # Divide by 2.0 because of the YOLO weight

    if avg_confidence > threshold:
        return best_product, avg_confidence
    else:
        return "Low confidence detection", avg_confidence

class ProductDetectionSystem:
    def __init__(self, model_path='yolo8n-l.pt', conf_threshold=0.25, ocr_threshold=0.3):
        self.detector = ObjectDetector(model_path, conf_threshold)
        self.ocr_processor = OCRProcessor()
        self.ocr_threshold = ocr_threshold

    def process_frame(self, frame):
        logging.info("Processing frame")

        yolo_detections = self.detector.detect(frame)
        ocr_results = self.ocr_processor.perform_ocr(frame)

        ocr_detections = []
        for bbox, text, conf in ocr_results:
            if conf > self.ocr_threshold:
                extracted_text, text_conf = self.ocr_processor.extract_text([(bbox, text, conf)])
                ocr_detections.append({
                    'bbox': bbox,
                    'class': extracted_text,
                    'conf': text_conf,
                    'detection_type': 'OCR'
                })

        all_detections = yolo_detections + ocr_detections

        # Draw bounding boxes and labels
        for detection in all_detections:
            bbox = detection['bbox']
            if isinstance(bbox, (tuple, list)):
                if len(bbox) == 4 and all(isinstance(coord, (int, float)) for coord in bbox):
                    x1, y1, x2, y2 = map(int, bbox)
                elif len(bbox) == 4 and all(isinstance(point, list) and len(point) == 2 for point in bbox):
                    x1 = int(min(point[0] for point in bbox))
                    y1 = int(min(point[1] for point in bbox))
                    x2 = int(max(point[0] for point in bbox))
                    y2 = int(max(point[1] for point in bbox))
                else:
                    logging.warning(f"Unexpected bbox format: {bbox}")
                    continue
            else:
                logging.warning(f"Unexpected bbox format: {bbox}")
                continue

            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{detection['class']} ({detection['conf']:.2f})"
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        return frame, all_detections

def process_stream(system, cap, stream_type):
    frame_count = 0
    all_detections = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1
        logging.info(f"Processing {stream_type} frame {frame_count}")

        processed_frame, frame_detections = system.process_frame(frame)
        all_detections.extend(frame_detections)

        cv2.imshow(f'Processed {stream_type}', processed_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            logging.info(f"{stream_type} processing stopped by user")
            break

    cap.release()
    cv2.destroyAllWindows()
    logging.info(f"{stream_type} processing complete. Total frames processed: {frame_count}")
    logging.info(f"Total detections accumulated: {len(all_detections)}")

    return all_detections

# def process_video(system, video_path):
#     logging.info(f"Processing video: {video_path}")
#     cap = cv2.VideoCapture(video_path)
#     if not cap.isOpened():
#         logging.error(f"Error opening video file: {video_path}")
#         return []

#     frame_count = 0
#     all_detections = []

#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break

#         frame_count += 1
#         logging.info(f"Processing frame {frame_count}")

#         processed_frame, frame_detections = system.process_frame(frame)
#         all_detections.extend(frame_detections)  # Accumulate all detections

#         cv2.imshow('Processed Video', processed_frame)
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             logging.info("Video processing stopped by user")
#             break

#     cap.release()
#     cv2.destroyAllWindows()
#     logging.info(f"Video processing complete. Total frames processed: {frame_count}")
#     logging.info(f"Total detections accumulated: {len(all_detections)}")

#     return all_detections

# def main():
#     logging.info("Starting Improved Product Detection System")
#     try:
#         system = ProductDetectionSystem(conf_threshold=0.25, ocr_threshold=0.3)
#         video_path = 'data/test10.mp4'  # Replace with your video path
#         all_detections = process_video(system, video_path)

#         logging.info(f"All detections: {all_detections}")  # Add this line for debugging

#         # Filter out any non-dictionary elements and separate YOLO and OCR detections
#         valid_detections = [d for d in all_detections if isinstance(d, dict)]
#         yolo_detections = [d for d in valid_detections if d.get('detection_type') == 'YOLO']
#         ocr_detections = [d for d in valid_detections if d.get('detection_type') == 'OCR']

#         logging.info(f"Total valid detections: {len(valid_detections)}")
#         logging.info(f"Total YOLO detections: {len(yolo_detections)}")
#         logging.info(f"Total OCR detections: {len(ocr_detections)}")

#         # Log a sample of detections for debugging
#         logging.info(f"Sample YOLO detection: {yolo_detections[:1] if yolo_detections else 'None'}")
#         logging.info(f"Sample OCR detection: {ocr_detections[:1] if ocr_detections else 'None'}")

#         # Get the final result
#         final_product, final_confidence = get_final_product(valid_detections, [], threshold=0.5)

#         print(f"\nFinal Result:")
#         print(f"Product: {final_product}")
#         print(f"Confidence: {final_confidence:.2f}")

#     except Exception as e:
#         logging.error(f"An error occurred in the main execution: {e}")
#         logging.error(f"Error details: {type(e).__name__}, {str(e)}")
#         import traceback
#         logging.error(f"Traceback: {traceback.format_exc()}")
#         print(f"Error: {e}")

# if __name__ == "__main__":
#     main()

def process_video(system, video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        logging.error(f"Error opening video file: {video_path}")
        return []

    return process_stream(system, cap, "Video")

def process_webcam(system):
    cap = cv2.VideoCapture(0)  # 0 is usually the default webcam
    if not cap.isOpened():
        logging.error("Error opening webcam")
        return []

    return process_stream(system, cap, "Webcam")

def main():
    logging.info("Starting Improved Product Detection System")
    try:
        system = ProductDetectionSystem(conf_threshold=0.25, ocr_threshold=0.3)

        # Choose between video file and webcam
        use_webcam = input("Use webcam? (y/n): ").lower() == 'y'

        if use_webcam:
            all_detections = process_webcam(system)
        else:
            video_path = 'data/test10.mp4'  # Replace with your video path
            all_detections = process_video(system, video_path)

        logging.info(f"All detections: {all_detections}")

        valid_detections = [d for d in all_detections if isinstance(d, dict)]
        yolo_detections = [d for d in valid_detections if d.get('detection_type') == 'YOLO']
        ocr_detections = [d for d in valid_detections if d.get('detection_type') == 'OCR']

        logging.info(f"Total valid detections: {len(valid_detections)}")
        logging.info(f"Total YOLO detections: {len(yolo_detections)}")
        logging.info(f"Total OCR detections: {len(ocr_detections)}")

        logging.info(f"Sample YOLO detection: {yolo_detections[:1] if yolo_detections else 'None'}")
        logging.info(f"Sample OCR detection: {ocr_detections[:1] if ocr_detections else 'None'}")

        final_product, final_confidence = get_final_product(valid_detections, [], threshold=0.5)

        print(f"\nFinal Result:")
        print(f"Product: {final_product}")
        print(f"Confidence: {final_confidence:.2f}")

    except Exception as e:
        logging.error(f"An error occurred in the main execution: {e}")
        logging.error(f"Error details: {type(e).__name__}, {str(e)}")
        import traceback
        logging.error(f"Traceback: {traceback.format_exc()}")
        print(f"Error: {e}")

if __name__ == "__main__":
    main()