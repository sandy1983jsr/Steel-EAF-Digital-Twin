from fastapi import FastAPI
from blast_furnace_model import BlastFurnaceSimulator, detect_anomaly

app = FastAPI()
model = BlastFurnaceSimulator()

@app.get("/")
def root():
    return {"msg": "Blast Furnace Digital Twin API"}

@app.post("/predict/")
def predict(data: dict):
    prediction = model.simulate_step(data)
    anomaly = detect_anomaly(data)
    return {"prediction": prediction, "anomaly": anomaly}
