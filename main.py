import os
import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
from PIL import Image
import io

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the model
model = tf.keras.models.load_model('model/animelens_model.h5')

# Define image size
IMAGE_SIZE = (224, 224)

def preprocess_image(image):
    try:
        # Convert to PIL Image
        if isinstance(image, np.ndarray):
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Resize using PIL with BILINEAR interpolation
        image = image.resize(IMAGE_SIZE, Image.BILINEAR)
        
        # Convert back to numpy array
        image = np.array(image)
        
        # Normalize pixel values
        image = image.astype(np.float32) / 255.0
        
        # Add batch dimension
        image = np.expand_dims(image, axis=0)
        
        return image
    except Exception as e:
        print(f"Error in preprocessing: {str(e)}")
        raise

@app.get("/")
async def root():
    return {"message": "Welcome to AnimeLens API"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        # Read and decode the image
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            return {"success": False, "error": "Invalid image file"}
        
        # Preprocess the image
        processed_image = preprocess_image(image)
        
        # Make prediction
        predictions = model.predict(processed_image, verbose=0)
        
        # Get all predictions sorted by confidence
        # Order matches the training data class indices
        class_labels = [
            'Hello World',
            'Josee, the Tiger and the Fish',
            'Natsu e no Tunnel Sayonara no Deguchi',
            'The Garden of Words',
            'Your Name'
        ]
        
        # Create list of predictions with movie names and confidence scores
        predictions_list = []
        for i, conf in enumerate(predictions[0]):
            predictions_list.append({
                "movie": class_labels[i],
                "confidence": float(conf)
            })
        
        # Sort predictions by confidence in descending order
        predictions_list.sort(key=lambda x: x["confidence"], reverse=True)
        
        return {
            "success": True,
            "predictions": predictions_list
        }
        
    except Exception as e:
        print(f"Error in prediction: {str(e)}")
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
