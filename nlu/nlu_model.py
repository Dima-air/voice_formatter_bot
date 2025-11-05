# nlu/nlu_model.py
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import re

class NLUModel:
    def __init__(self, model_path="./nlu_model"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_path)
        self.model.eval()
        self.patterns = {
            "format_bold": r"(?:жирным|жирный|выдели жирным)\s+(?:слово|текст|фразу|название)?\s*([^\.,!?]+)",
            "format_italic": r"(?:курсивом|курсив|наклонным|наклонным шрифтом)\s+(?:слово|текст|фразу|выражение)?\s*([^\.,!?]+)",
            "format_header": r"(?:заголовок|заголовком|заголовке)\s+(?:для|второго уровня|надпись)?\s*([^\.,!?]+)",
        }

    def predict(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True)
        with torch.no_grad():
            logits = self.model(**inputs).logits
        intent_id = torch.argmax(logits, dim=-1).item()
        intent = self.model.config.id2label[intent_id]
        match = re.search(self.patterns.get(intent, r"(.+)"), text, re.IGNORECASE)
        entity = match.group(1).strip() if match else text
        return intent, entity