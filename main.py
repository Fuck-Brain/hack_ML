from typing import List
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import numpy as np
from operator import itemgetter
import json

app = FastAPI()

labels = ["человек, который ищёт друзей или единомышленников", 
          "человек, который ищет сотрудника или исполнителя",
          "человек, который оказывает или продаёт услуги",
          "человек, который ищет работу или проект",
          "человек, который хочет получить услугу"]

model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
classifier = pipeline("zero-shot-classification",
                      model="MoritzLaurer/mDeBERTa-v3-base-mnli-xnli")


with open("themes.json", "r", encoding="utf-8") as f:
    themes = json.load(f)

themes_emb = model.encode(
        themes,
        convert_to_numpy=True,
        normalize_embeddings=True,   
        batch_size=128,
        show_progress_bar=False
    ) #[model.encode(theme) for theme in themes]

class User(BaseModel):
    Id: str
    DescribeUser: str
    Skills: str
    Interests: str
    Hobbies: str

    def getText(self) ->str:
        return ("О себе: " + self.DescribeUser + 
        ". Мои навыки: " + self.Skills + 
        ". Мои интересы: " + self.Interests +
        ". Мои Хобби: " + self.Hobbies)

class MyRequest(BaseModel):
    UserId: str
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
    Users: List[User]
    Requests: List[MyRequest]

@app.post("/classifier")
async def classification(request: MyRequest):
    print("start classification")
    data = request.getText()

    scores = classifier(data, labels)
    result = scores["labels"][0]
    return JSONResponse(content=result)

def cosine_similarity(A: np.ndarray, B: np.ndarray) -> float:
    dot_product = np.dot(A, B)
    norm_A = np.linalg.norm(A)
    norm_B = np.linalg.norm(B)
    return dot_product / (norm_A * norm_B)

def check_label(main_label: str, label: str) -> bool:
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
    print("start predict")
    request = model.encode(request_body.Request.getText())
    main_label = request_body.Request.Label

    #пользователи без запросов (вначале все, в ходе обработки запросов лишние удаляются)
    user_dict = {user.Id: user for user in request_body.Users}

    request_scores = {} #сходство по запросам ( ключ - коэф. схожести, значение - id пользователя)
    user_scores = {} #сходство по профилям (без запросов)

    for r in request_body.Requests:
        
        if check_label(main_label, r.Label): 
            score = cosine_similarity(model.encode(r.getText()), request)
            if (r.UserId in request_scores.keys()):
                request_scores[r.UserId] = max(request_scores[r.UserId], score)
            else: request_scores[r.UserId] = score
        elif (main_label != labels[3]): 
            score = cosine_similarity(model.encode(user_dict[r.UserId].getText()), request)
            if (r.UserId in user_scores.keys()):
                user_scores[r.UserId] = max(user_scores[r.UserId], score)
            else:
                user_scores[r.UserId] = score
        user_dict.pop(r.UserId, None)

    #Проверяются оставшиеся поьзователи без запросов 
    for key in user_dict.keys():
        score = cosine_similarity(model.encode(user_dict[key].getText()), request)
        user_scores[key] = score

    request_scores = dict(sorted(request_scores.items(), key=itemgetter(1), reverse=True))
    user_scores = dict(sorted(user_scores.items(), key=itemgetter(1), reverse=True))

    request_scores = {k: float(v) for k, v in request_scores.items()}
    user_scores    = {k: float(v) for k, v in user_scores.items()}

    return JSONResponse(content = {**request_scores , **user_scores})

def count_theme(request: List[str]):
    count_theme = {}

    skills = []
    for s in request:
        for skill in s.split(','):
            skills.append(skill)

    skill_emb = model.encode(
        skills,
        convert_to_numpy=True,
        normalize_embeddings=True,
        batch_size=256,
        show_progress_bar=False
    )
    similarity = skill_emb @ themes_emb.T
    idx = np.argmax(similarity, axis=1)

    counts = np.bincount(idx, minlength=themes_emb.shape[0])

    # Формируем ответ, отсортировав по убыванию
    order = np.argsort(-counts)
    count_theme = {themes[i].split(":")[0]: int(counts[i]) for i in order if counts[i] > 0}
    return count_theme

#упорядачивает навыки(хобби/интересы) по популярности среди пользователей (поиск по анкетам)
@app.post("/statistic/most_popular")
async def most_popular(request: List[str]):
    print("start finding most_popular")
    return JSONResponse(content = count_theme(request))

class RequestBody1(BaseModel):
    Skills: List[str]
    Requests: List[MyRequest]

#упорядачивает навыки(хобби/интересы) по востребованности в запросах (с учётом их меток)
@app.post("/statistic/requests_frequency")
async def requests_frequency(request: RequestBody1):
    print("start finding frequency in requests")
    accuracy = 0.35

    filtered_requests = [req for req in request.Requests 
                         if req.Label in {labels[0], labels[1], labels[4]}]
    
    
    
    texts = [request.getText() for request in filtered_requests]
    
    texts_emb = model.encode(texts, 
                             convert_to_numpy=True, 
                             normalize_embeddings=True, 
                             batch_size=128)

    similarity = texts_emb @ themes_emb.T    
        
    counts = (similarity >= accuracy).sum(axis=0)  

    order = np.argsort(-counts)
    skills_count = {themes[i].split(":")[0]: int(counts[i]) for i in order if counts[i] > 0}

    return JSONResponse(content=skills_count)







    