#src/ocr_processor.py
import easyocr
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import logging
from collections import Counter
from difflib import SequenceMatcher
import heapq


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

    def extract_text(self, ocr_results):
        try:
            extracted_text = " ".join([text for _, text, _ in ocr_results])
            words = extracted_text.split()

            for i, word in enumerate(words):
                if '[UNK]' in word or len(word) < 3:
                    masked_text = " ".join(words[:i] + ['[MASK]'] + words[i+1:])
                    predictions = self.bert_predict(masked_text)
                    best_prediction = self.get_close_matches(word, predictions, n=1, cutoff=0.6)
                    if best_prediction:
                        words[i] = best_prediction[0]

            corrected_text = " ".join(words)
            return corrected_text, 1.0  # Returning confidence as 1.0 for simplicity
        except Exception as e:
            logging.error(f"Text extraction failed: {e}")
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

    @staticmethod
    def get_close_matches(word, possibilities, n=3, cutoff=0.6):
        if not n > 0:
            raise ValueError("n must be > 0: %r" % (n,))
        if not 0.0 <= cutoff <= 1.0:
            raise ValueError("cutoff must be in [0.0, 1.0]: %r" % (cutoff,))
        result = []
        s = SequenceMatcher()
        s.set_seq2(word.lower())
        for x in possibilities:
            s.set_seq1(x.lower())
            if s.real_quick_ratio() >= cutoff and \
               s.quick_ratio() >= cutoff and \
               s.ratio() >= cutoff:
                result.append((s.ratio(), x))
        # Move the best scorers to head of list
        result = heapq.nlargest(n, result)
        # Strip scores for the best n matches
        return [x for score, x in result]