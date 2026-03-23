import pickle
import joblib
import os
import logging

BASE = os.path.dirname(__file__)

def load_models():
    models, scalers = {}, {}
    files = {
        "diabetes": ("diabetes_model.sav", "diabetes_scaler.sav"),
        "stroke":   ("stroke_model.pkl",   None),
        "heart":    ("heart_model.pkl",     None),
    }
    for name, (model_file, scaler_file) in files.items():
        try:
            filepath = os.path.join(BASE, model_file)
            try:
                models[name] = joblib.load(filepath)
            except Exception:
                with open(filepath, "rb") as f:
                    models[name] = pickle.load(f)
            logging.info(f"✅ Loaded {name} model")
        except FileNotFoundError:
            logging.warning(f"⚠️  {name} model not found — skipping")
            models[name] = None
        except Exception as e:
            logging.error(f"❌ Error loading {name} model: {e}")
            models[name] = None

        if scaler_file:
            try:
                scalerpath = os.path.join(BASE, scaler_file)
                try:
                    scalers[name] = joblib.load(scalerpath)
                except Exception:
                    with open(scalerpath, "rb") as f:
                        scalers[name] = pickle.load(f)
            except FileNotFoundError:
                scalers[name] = None
            except Exception as e:
                logging.error(f"❌ Error loading {name} scaler: {e}")
                scalers[name] = None

    return models, scalers
