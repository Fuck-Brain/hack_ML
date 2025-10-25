from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from transformers import pipeline
from sentence_transformers import SentenceTransformer
from pydantic import BaseModel
import numpy as np
from operator import itemgetter
from collections import Counter
from rapidfuzz.fuzz import partial_ratio

app = FastAPI()

labels = ["человек, который ищёт друзей или единомышленников", 
          "человек, который ищет сотрудника или исполнителя",
          "человек, который оказывает или продаёт услуги",
          "человек, который ищет работу или проект",
          "человек, который хочет получить услугу"]

model = SentenceTransformer('sentence-transformers/distiluse-base-multilingual-cased-v2')
classifier = pipeline("zero-shot-classification",
                      model="joeddav/xlm-roberta-large-xnli")

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
    Users: list[User]
    Requests: list[MyRequest]

@app.post("/classifier")
async def classification(request: MyRequest):
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
    request = model.encode(request_body.Request.getText())
    main_label = request_body.Request.Label

    #пользователи без запросов (вначале все, в ходе обработки запросов лишние удаляются)
    user_dict = {user.Id: user for user in request_body.Users}

    request_scores = {} #сходство по запросам ( ключ - коэф. схожести, значение - id пользователя)
    user_scores = {} #сходство по профилям (без запросов)

    for r in request_body.Requests:
        user_dict.pop(r.UserId, None)
        
        if check_label(main_label, r.Label): 
            score = cosine_similarity(model.encode(r.getText()), request)
            if (r.UserId in request_scores.keys()):
                request_scores[r.UserId] = max(request_scores[r.UserId], score)
            else: request_scores[r.UserId] = score
        elif (main_label != labels[3]): 
            score = cosine_similarity(model.encode(r.User.getText()), request)
            if (r.UserId in user_scores.keys()):
                user_scores[r.UserId] = max(user_scores[r.UserId], score)
            else:
                user_scores[r.UserId] = score

    #Проверяются оставшиеся поьзователи без запросов 
    for key in user_dict.keys():
        score = cosine_similarity(model.encode(user_dict[key].getText()), request)
        user_scores[key] = score

    request_scores = dict(sorted(request_scores.items(), key=itemgetter(1), reverse=True))
    user_scores = dict(sorted(user_scores.items(), key=itemgetter(1), reverse=True))

    return JSONResponse(content = {**request_scores , **user_scores})


def count_words(request: list[str]):
    terms = []
    # прведение к нижнему регистру
    for s in request:
        for word in s.split(','):
            terms.append(word.lower().strip())

    freq = Counter(terms)

    emb = model.encode(terms)

    # граф близости с гвардами
    n = len(terms)
    edges = [[] for _ in range(n)]
    tau_sem = 0.7
    tau_str = 70  # RapidFuzz 0..100

    for i in range(n):
        for j in range(i+1, n):
            sem = float(np.dot(emb[i], emb[j]))
            if sem >= tau_sem:
                if partial_ratio(terms[i], terms[j]) >= tau_str:
                    edges[i].append(j)
                    edges[j].append(i)

    # компоненты связности (объединяем леммы)
    visited = [False]*n
    clusters = []
    for i in range(n):
        if visited[i]: continue
        stack = [i]; comp = []
        visited[i] = True
        while stack:
            k = stack.pop()
            comp.append(k)
            for nb in edges[k]:
                if not visited[nb]:
                    visited[nb] = True
                    stack.append(nb)
        clusters.append(comp)
        
    count_words = []
    for comp in clusters:
        aliases = [terms[i] for i in comp]
        total = sum(freq[a] for a in aliases)
        canonical = max(aliases, key=lambda a: freq[a])  # канон = самый частотный
        count_words.append({"interest": canonical, "count": total})
    
    count_words.sort(key = lambda x: x["count"], reverse=True)
    return count_words

#упорядачивает навыки(хобби/интересы) по популярности среди пользователей (поиск по анкетам)
@app.post("/statistic/most_popular")
async def most_popular(request: list[str]):
    return JSONResponse(content = count_words(request))

class RequestBody1(BaseModel):
    Skills: list[str]
    Requests: list[MyRequest]

#упорядачивает навыки(хобби/интересы) по востребованности в запросах (с учётом их меток)
@app.post("/statistic/requests_frequency")
async def requests_frequency(request: RequestBody1):
    labels = ["Здесь требуется знание (умение/навык) " + item["interest"] for item in count_words(request.Skills)]
    skills_count = {} #dict: skill - count
    accuracy = 0.7

    filtered_requests = [req for req in request.Requests 
                         if req.Label in {labels[0], labels[1], labels[4]}]

    for request in filtered_requests:
        text = request.getText()
        result = classifier(text, candidate_labels=labels, multi_label=True)
        for i in range(len(request.Skills)):
            if (result["scores"][i] >= accuracy):
                skill = result["labels"][i].split(' ')[-1]
                skills_count[skill] = skills_count.get(skill, 0) + 1

    return JSONResponse(content=skills_count)







    