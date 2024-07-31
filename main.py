from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import tensorflow as tf
import pandas as pd

app = FastAPI()

# Load the TensorFlow model
try:
    model = tf.keras.models.load_model('3epoches.h5')
except Exception as e:
    raise RuntimeError(f"Error loading model: {e}")

# Load the TextVectorization layer
try:
    vectorizer_model = tf.keras.models.load_model('vectorizer_model')
    vectorizer = vectorizer_model.layers[0]  # Assuming the TextVectorization layer is the first layer
except Exception as e:
    raise RuntimeError(f"Error loading vectorizer: {e}")

# Define the actual column names used by your model
column_names = ['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']

# Define the request body structure
class PredictionRequest(BaseModel):
    text: str

# Define the comment_score function
def comment_score(comment):
    vectorized_comment = vectorizer([comment])  # Use the loaded vectorizer to transform the comment
    results = model.predict(vectorized_comment)
    
    print("Vectorized Comment: ", vectorized_comment)
    print("Prediction Results: ", results)

    text = ''
    # Ensure the loop covers all columns of the results
    for idx, col in enumerate(column_names):
        text += '{}: {}\n'.format(col, results[0][idx] > 0.5)

    print("Formatted Prediction Text: ", text)
    return text

# Create a prediction endpoint
@app.post("/predict")
def predict(request: PredictionRequest):
    try:
        # Get the prediction results
        prediction_text = comment_score(request.text)

        # Return the predictions
        return {"predictions": prediction_text}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
