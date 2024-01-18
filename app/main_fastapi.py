import sys
from pathlib import Path
from fastapi import FastAPI, HTTPException

sys.path.append(str(Path(__file__).resolve().parent.parent))

from src.predict_model import predict
from src.load_models import load_bucket

app = FastAPI()

@app.get("/")
async def info():
    try:
        return {"Endpoints": 
                {
                    "/predictions/": "Call that endpoint with a 'data' argument to get a prediction",
                    "/update_model/": "Call that endpoint to update to the latest best model"
                }
                }
    except Exception as e:
        # Handle exceptions
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/predictions/")
async def make_prediction(data: str):
    try:
        prediction_result = predict([data], which_model="best")
        prediction = prediction_result[0]

        response = "CATASTROPHE!" if prediction == 1 else "Relax, no catastrophe."
        return {"sentence": data, "prediction": response}

    except Exception as e:
        # Handle exceptions
        raise HTTPException(status_code=500, detail=str(e))
    
@app.get("/update_model/")
async def update_model():
    try:
        load_bucket("models")
        return {"result": "Updated model!"}

    except Exception as e:
        # Handle exceptions
        raise HTTPException(status_code=500, detail=str(e))
