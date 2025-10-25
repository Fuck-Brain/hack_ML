from fastapi import FastAPI
from fastapi.responses import JSONResponse
from transformers import pipline
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import numpy as np
from operator import itemgetter

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
    User: User
    NameRequest: str
    TextRequest: str
    Label: str
    
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

def check_label(main_label, label):
    #1:работодатель - работник 
    #2:продавец - покупатель 
    #3:работник - работодатель
    #4:покупатель - продавец
    return ((main_label == labels[1] and label == labels[3]) or 
            (main_label == labels[2] and label == labels[4]) or 
            (main_label == labels[3] and label == labels[1]) or 
            (main_label == labels[4] and label == labels[2]))   


@app.post("/predict")
async def predict(request_body: RequestBody):
    request = model.encode(request_body.Request.getText())
    main_label = request_body.Request.Label

    #пользователи без запросов (вначале все, в ходе обработки запросов лишние удаляются)
    user_dict = {user.Id: user for user in request_body.Users}

    request_scores = {} #сходство по запросам ( ключ - коэф. схожести, значение - id пользователя)
    user_scores = {} #сходство по профилям (без запросов)

    for r in request_body.Requests:
        user_dict.pop(r.UserId, None)
        
        if check_label(main_label, r.Label): 
            score = cosine_similary(model.encode(r.getText()), request)
            if (r.UserId in request_scores.keys()):
                request_scores[r.UserId] = max(request_scores[r.UserId], score)
            else: request_scores[r.UserId] = score
        elif (main_label != labels[3]): 
            score = cosine_similary(model.encode(r.User.getText()), request)
            if (r.UserId in user_scores.keys()):
                user_scores[r.UserId] = max(user_scores[r.UserId], score)
            else:
                user_scores[r.UserId] = score

    #Проверяются оставшиеся поьзователи без запросов 
    for key in user_dict.keys():
        score = cosine_similary(model.encode(user_dict[key].getText()), request)
        user_scores[key] = score

    request_scores = dict(sorted(request_scores.items(), key=itemgetter(1)))
    user_scores = dict(sorted(user_scores.items(), key=itemgetter(1)))

    return JSONResponse(content = request_scores | user_scores)






    