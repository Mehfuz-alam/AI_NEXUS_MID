from fastapi import FastAPI,File,UploadFile
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from io import BytesIO

app=FastAPI()

# Load model and processor once at startup
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

@app.post("/generate-caption/")
async def generate_caption(file: UploadFile= File(...)):
    try:
        #Read the image
        image_data=await file.read()
        image=Image.open(BytesIO(image_data))

        # Process the image and generate the caption
        inputs=processor(image,return_tensors="pt")
        out=model.generate(**inputs)
        caption=processor.decode(out[0],skip_special_tokens=True)

        return {"caption":caption}

    except Exception as e:
        return {"error":str(e)}