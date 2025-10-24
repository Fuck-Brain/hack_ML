from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import (pipline)

app = FastAPI()

labels = ["человек, который ищёт друзей или единомышленников", 
          "человек, который ищет сотрудника или исполнителя",
          "человек, который оказывает или продаёт услуги",
          "человек, который ищет работу или проект",
          "человек, который хочет получить услугу"]

@app.post("/")
async def classification(request: Request):
    data = await request.json()
    
    classifier = pipline("zero-shot-classification",
                      model="joeddav/xlm-roberta-large-xnli")
    scores = classifier(data, labels)
    result = scores["labels"][0]
    return JSONResponse(content=result)
