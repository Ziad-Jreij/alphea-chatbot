# components/emotion.py
from __future__ import annotations
from typing import Any, Dict, List, Text
from rasa.engine.graph import GraphComponent
from rasa.engine.recipes.default_recipe import DefaultV1Recipe
from rasa.engine.storage.resource import Resource
from rasa.engine.storage.storage import ModelStorage
from rasa.shared.nlu.training_data.message import Message
from rasa.shared.nlu.constants import TEXT
from transformers import AutoTokenizer, AutoModelForSequenceClassification, TextClassificationPipeline
from threading import Lock
import torch

MODEL_NAME = "joeddav/distilbert-base-uncased-go-emotions-student"

_POS = {"admiration","amusement","approval","caring","gratitude","joy","love","optimism","pride","relief"}
_NEG = {"anger","annoyance","disappointment","disapproval","disgust","embarrassment","grief","fear","nervousness","remorse","sadness"}
_NEU = {"neutral"}

_init_lock = Lock()
_pipe: TextClassificationPipeline | None = None

def ensure_emotion_pipe() -> TextClassificationPipeline:
    global _pipe
    if _pipe is None:
        with _init_lock:
            if _pipe is None:
                device = 0 if torch.cuda.is_available() else -1
                tok = AutoTokenizer.from_pretrained(MODEL_NAME)
                mdl = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME)
                _pipe = TextClassificationPipeline(model=mdl, tokenizer=tok, top_k=None, function_to_apply="sigmoid", device=device)
    return _pipe

def coarse_from_label(emotion_label: str) -> str:
    e = (emotion_label or "").lower()
    if e in _POS: return "positive"
    if e in _NEG: return "negative"
    if e in _NEU: return "neutral"
    return "ambiguous"

def top_emotion(text: str) -> tuple[str, list[tuple[str, float]]]:
    pipe = ensure_emotion_pipe()
    preds = pipe((text or "").strip())[0]
    top = max(preds, key=lambda x: x["score"])
    top3 = sorted(((p["label"], float(p["score"])) for p in preds), key=lambda x: x[1], reverse=True)[:3]
    return top["label"], top3

@DefaultV1Recipe.register(component_types=[GraphComponent], is_trainable=False)
class EmotionClassifierComponent(GraphComponent):
    @staticmethod
    def get_default_config() -> Dict[Text, Any]:
        return {}

    @classmethod
    def create(cls, config: Dict[Text, Any], model_storage: ModelStorage, resource: Resource):
        return cls(config)

    def __init__(self, config: Dict[Text, Any]) -> None:
        self.config = config

    def train(self, training_data: Any) -> None:
        return

    def process(self, messages: List[Message]) -> List[Message]:
        for m in messages:
            text = (m.get(TEXT) or "")
            label, _top3 = top_emotion(text)
            m.set("top_emotion", label, add_to_output=True)
        return messages
