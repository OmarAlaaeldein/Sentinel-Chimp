"""Switchable transformer sentiment engine (FinBERT)."""
import os

try:
    import torch
    import torch.nn.functional as F
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    torch = None
    F = None
    AutoTokenizer = None
    AutoModelForSequenceClassification = None


class SentimentEngine:
    def __init__(self):
        self.current_model_name = "FinBERT"
        self.models = {
            "FinBERT": {
                "id": "ProsusAI/finbert",
                "dir": "my_finbert_model",
                "loaded": False,
                "tokenizer": None,
                "model": None,
                "pos_idx": 0,
                "neg_idx": 1,
            }
        }
        self.status_msg = "Initializing..."

    def load_model(self, model_key):
        if not TRANSFORMERS_AVAILABLE:
            self.status_msg = "Error: 'transformers' library missing."
            return False

        target = self.models[model_key]
        if target["loaded"]:
            self.current_model_name = model_key
            self.status_msg = f"{model_key} Ready (Cached)."
            return True

        local_path = os.path.join(os.getcwd(), target["dir"])
        
        # Determine device (CUDA if available)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        
        try:
            print(f"[System] Loading {model_key} on {device}...")
            if os.path.exists(local_path) and os.listdir(local_path):
                target["tokenizer"] = AutoTokenizer.from_pretrained(local_path)
                target["model"] = AutoModelForSequenceClassification.from_pretrained(local_path).to(device)
            else:
                print(f"[System] Downloading {model_key} (First Run)...")
                target["tokenizer"] = AutoTokenizer.from_pretrained(target["id"])
                target["model"] = AutoModelForSequenceClassification.from_pretrained(target["id"]).to(device)
                
                print(f"[System] Saving {model_key} locally...")
                target["tokenizer"].save_pretrained(local_path)
                target["model"].save_pretrained(local_path)
            
            # Build label index map from id2label (don't assume hardcoded order).
            id2label = getattr(target["model"].config, 'id2label', None)
            if id2label:
                label2id = {str(v).lower(): int(k) for k, v in id2label.items()}
                if 'positive' not in label2id or 'negative' not in label2id:
                    raise RuntimeError(
                        f"Unexpected FinBERT labels for {model_key}: {id2label}. "
                        f"Expected labels containing 'positive' and 'negative'."
                    )
                target["pos_idx"] = label2id['positive']
                target["neg_idx"] = label2id['negative']

            # Inference-only: disable dropout and avoid autograd bookkeeping.
            target["model"].eval()

            # Only mark loaded AFTER successful validation so a failure doesn't
            # leave the model silently usable with a wrong label mapping.
            target["loaded"] = True
            self.current_model_name = model_key

            self.status_msg = f"{model_key} Loaded ({device.upper()})."
            return True

        except Exception as e:
            self.status_msg = f"Failed to load {model_key}: {e}"
            print(f"[Error] {self.status_msg}")
            return False

    def predict_batch(self, texts):
        target = self.models[self.current_model_name]
        
        # Use "Pending" instead of 0.5 for non-loaded models
        if not target["loaded"]:
            return ["Pending" for _ in texts]
            
        clean_texts = [t for t in texts if t and t.strip()]
        if not clean_texts:
            return ["Pending" for _ in texts]
            
        try:
            # Move inputs to the same device as the model
            device = next(target["model"].parameters()).device
            inputs = target["tokenizer"](clean_texts, return_tensors="pt", truncation=True,
                                          padding=True, max_length=128).to(device)
            
            with torch.no_grad():
                outputs = target["model"](**inputs)
            
            probs = F.softmax(outputs.logits, dim=-1)

            pos_idx = target.get("pos_idx", 0)
            neg_idx = target.get("neg_idx", 1)
            pos = probs[:, pos_idx]
            neg = probs[:, neg_idx]
            # Convert back to CPU for list conversion
            scores_clean = (0.5 + (pos * 0.5) - (neg * 0.5)).cpu().tolist()
                
            full_scores = []
            idx = 0
            for t in texts:
                if t and t.strip():
                    full_scores.append(scores_clean[idx])
                    idx += 1
                else:
                    full_scores.append("Pending")
            return full_scores
        except Exception as e:
            print(f"[Model Error] {e}")
            return ["Pending" for _ in texts]



sentiment_engine = SentimentEngine()
