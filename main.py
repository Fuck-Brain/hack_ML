from fastapi import FastAPI
from fastapi.responses import JSONResponse
from transformers import pipline
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import numpy as np

app = FastAPI()

labels = ["человек, который ищёт друзей или единомышленников", 
          "человек, который ищет сотрудника или исполнителя",
          "человек, который оказывает или продаёт услуги",
          "человек, который ищет работу или проект",
          "человек, который хочет получить услугу"]

model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')

class User(BaseModel):
    Id: str
    DescribeUser: str
    Skills: list[str]
    Interests: list[str]
    Hobbies: list[str]

    def getText(self) ->str:
        return ("О себе: " + self.DescribeUser + 
        ". Мои навыки: " + ", ".join(self.Skills) + 
        ". Мои интересы: " + ", ".join(self.Interests) +
        ". Мои Хобби: " + ", ".join(self.Hobbies))
    

class MyRequest(BaseModel):
    UserId: str
    NameRequest: str
    TextRequest: str
    
    def getText(self) -> str:
        if (self.NameRequest.endswith('.')):
            return self.NameRequest + ' ' + self.TextRequest
        else: 
            return self.NameRequest + '. ' + self.TextRequest

class RequestBody(BaseModel):
    Request: MyRequest
    Users: list[User]
    Requests: list[MyRequest]


@app.post("/classifire")
async def classification(request: MyRequest):
    data = request.getText()

    classifier = pipline("zero-shot-classification",
                      model="joeddav/xlm-roberta-large-xnli")
    scores = classifier(data, labels)
    result = scores["labels"][0]
    return JSONResponse(content=result)

def cosine_similary(A, B):
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    return dot_product / (norm_A * norm_B)

@app.post("/predict")
async def predict(request_body: RequestBody):
    request = model.encode(request_body.Request.getText())
    requests_dict = {request.UserId: request for request in request_body.Requests}
    #пользователи без запросов
    user_dict = {user.Id: user for user in request_body.Users 
                 if user.Id not in requests_dict.keys()}

    request_scores = {
        "Id": [],
        "Score": [] } #сходство по запросам (ключ - id пользователя, значение - коэф. схожести)
    user_scores = {
        "Id": [],
        "Score": []
    } #сходство по профилям (без запросов)

    for key in requests_dict.keys():
        request_scores[key] = 



    