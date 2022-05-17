from io import BytesIO
from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
from PIL import Image
import tensorflow as tf

app=FastAPI()
MODEL = tf.keras.models.load_model("my_model.h5")
class_predictions=['outdoor' , 'indoor']

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image


@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    image = read_file_as_image(await file.read())
    image.resize(50,50)
    img_batch = np.expand_dims(image, 0)
    img_batch = np.asarray(img_batch) / 255.0
    img_batch=img_batch.reshape(-1,50,50,1)

    predictions = MODEL.predict(img_batch)
    print(predictions)
    class_prediction = class_predictions[np.argmax(predictions)]
    return class_prediction


if __name__ == '__main__':

    uvicorn.run(app)
