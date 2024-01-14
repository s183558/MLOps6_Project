import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.predict_model import predict

app = FastAPI()

@app.get("/predictions/")
async def make_prediction(data: str):
    try:
        prediction_result = predict([data])
        prediction = prediction_result[0]

        response = "CATASTROPHE!" if prediction == 1 else "Relax, no catastrophe."
        return {"sentence": data, "prediction": response}

    except Exception as e:
        # Handle exceptions
        raise HTTPException(status_code=500, detail=str(e))
