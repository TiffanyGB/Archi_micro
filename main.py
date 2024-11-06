from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from appel_model import predict_wine_quality
import pickle

app = FastAPI()

class WineData(BaseModel):
    fixed_acidity: float
    volatile_acidity: float
    citric_acid: float
    residual_sugar: float
    chlorides: float
    free_sulfur_dioxide: float
    total_sulfur_dioxide: float
    density: float
    pH: float
    sulphates: float
    alcohol: float
    quality: int
    Id: int
    
@app.get("/")
async def root():
    return {"message": "Hello World"}

@app.post("/api/predict")
async def predict(data: WineData):
    try:
        prediction = predict_wine_quality(data.dict())
        return {"prediction": prediction}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# @app.get("/api/perfect_wine")
# async def get_perfect_wine():
#     try:
#         perfect_wine = find_perfect_wine()
#         return {"perfect_wine": perfect_wine}
#     except Exception as e:
#         raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, debug=True)