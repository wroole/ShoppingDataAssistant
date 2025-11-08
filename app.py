import openai
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import os, sqlite3, json, re, hashlib
import pandas as pd
from pydantic import BaseModel
from dotenv import load_dotenv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

load_dotenv()

OPENAI_API_KEY = os.getenv("KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY is not in environment")

openai.api_key = OPENAI_API_KEY

def load_prompt(path):
    with open(path, "r", encoding='utf-8') as f:
        return f.read()

SQL_PROMPT = load_prompt("prompts/to_sql.txt")
ANSWER_PROMPT = load_prompt("prompts/to_answer.txt")
CHART_PROMPT = load_prompt("prompts/to_chart.txt")

def generate_sql(question: str) -> str:
    message = [
        {'role': 'system', 'content': SQL_PROMPT},
        {'role': 'user', 'content': question}
    ]
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages = message,
        temperature = 0,
        max_tokens=300
    )
    sql_raw = response.choices[0].message.content.strip()
    return sql_raw

DISALLOWED = re.compile(r'\b(INSERT|UPDATE|DELETE|DROP|ALTER|CREATE|ATTACH|PRAGMA|VACUUM|EXEC|CALL)\b', re.I)

def validate_sql(sql: str) -> bool:
    if not sql:
        return False
    s = sql.strip()
    if s.upper() == 'NO_SQL_POSSIBLE' or s.upper() == 'SMALL_TALK':
        return True
    if not s.upper().startswith('SELECT'):
        return False
    if DISALLOWED.search(s):
        return False
    return True

GRAPH_DETECTION = re.compile(
    r"\b("
    r"chart|graph|plot|visualiz\w*|plotting|bar|line|pie|histogram|scatter|"
    r"графік|діаграм\w*|побудуй|намалю\w*|відобраз\w*|покаж\w*|"
    r"graf\w*|diagram\w*|nakresl\w*|vykresl\w*|zobraz\w*|vizualiz\w*"
    r")\b",
    re.I | re.U
)

def detect_graph_request(question: str) -> bool:
    return bool(GRAPH_DETECTION.search(question or ""))

STATIC_DIR = 'static'
CHART_DIR = os.path.join(STATIC_DIR, 'charts')
os.makedirs(CHART_DIR, exist_ok=True)

def generate_chart(question: str, sql: str, df: pd.DataFrame) -> dict | None:
    print("Generating chart spec...")
    cols = []
    for c in df.columns:
        dtype = str(df[c].dtype)
        sample = df[c].dropna().astype(str).head(5).tolist()
        cols.append({"name": c, "dtype": dtype, "sample": sample})
    context = {
        "question": question,
        "sql": sql,
        "columns": cols
    }
    message = [
        {'role': 'system', 'content': CHART_PROMPT},
        {'role': 'user', 'content': json.dumps(context, ensure_ascii=False)}
    ]
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages = message,
        temperature = 0,
        max_tokens=300
    )
    raw = response.choices[0].message.content.strip()
    if raw.startswith("```") and raw.endswith("```"):
        raw = '\n'.join(raw.split('\n')[1:-1]).strip()
    print("Chart spec raw:", raw)
    try:
        return json.loads(raw)
    except:
        return None
    
def generate_chart_image(df: pd.DataFrame, spec: dict) -> str | None:
    print("Generating chart image...")
    payload = {
        "spec": spec,
        "data": df.head(500).to_dict(orient="records")
    }
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, default=str).encode()).hexdigest()[:10]
    filename = f"chart_{digest}.png"
    path = os.path.join(CHART_DIR, filename)
    url = f"/{STATIC_DIR}/charts/{filename}"
    if os.path.exists(path):
        return url
    chart_type = (spec.get("type") or "count").lower()
    x = spec.get("x")
    y = spec.get("y")
    hue = spec.get("groupBy")
    if not x or x not in df.columns:
        return None
    if chart_type in ("bar", "column") and (not y or y not in df.columns):
        return None
    
    df = df.dropna(subset=[x] + ([y] if y else []))
    if df.empty:
        return None
    df = df.head(25)
    plt.figure(figsize=(10,6))
    plt.title(spec.get("title", "Chart"))
    plt.xticks(rotation=90)
    try:
        if chart_type == 'count':
            sns.countplot(data=df, x=x, hue=hue)
        elif chart_type == 'bar':
            sns.barplot(data=df, x=x, y=y, hue=hue, estimator=sum, errorbar=None)
        elif chart_type == 'line':
            if hue and hue in df.columns:
                for key, grp in df.groupby(hue):
                    plt.plot(grp[x], grp[y], marker='o', label=str(key))
                plt.legend()
            else:
                plt.plot(df[x], df[y], marker='o')
        else:
            print(f"Unsupported chart type: {chart_type}")
            return None
    except Exception as e:
        print(f"Error generating chart: {e}")
        return None
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    return url

def question_sql_answer(question: str, need_chart: bool) -> dict:
    sql = generate_sql(question)
    if not validate_sql(sql):
        return generate_answer({
                "question": question,
                "sql": 'DANGEROUS_MASSAGE',
                "result": [],
                "image": None,
                "error": "Dangerous SQL query"
            })
    elif sql.strip() == 'NO_SQL_POSSIBLE':
        return generate_answer({
            "question": question,
            "sql": "NO_SQL_POSSIBLE",
            "result": [],
            "image": None
        })
    elif sql.strip() == 'SMALL_TALK':
        return generate_answer({
            "question": question,
            "sql": 'SMALL_TALK',
            "result": [],
            "image": None
        })
    else:
        try:
            with sqlite3.connect('sales.db') as conn:
                df_res = pd.read_sql_query(sql, conn)
        except Exception as e:
            return generate_answer({
                "question": question,
                "sql": sql,
                "result": [],
                "image": None,
                "error": f"While executing SQL: {e}"
            })
        records = df_res.head(15).to_dict(orient="records")
        image_url = None
        if need_chart:
            try:
                chart_spec = generate_chart(question, sql, df_res)
                print("Chart spec:", chart_spec)
                if chart_spec:
                    image_url = generate_chart_image(df_res, chart_spec)
                    print("Generated chart URL:", image_url)
                else:
                    image_url = None
            except Exception as e:
                image_url = None
        answer = generate_answer({
            "question": question,
            "sql": sql,
            "result": records,
            "image": image_url
        })
        return answer
        

def generate_answer(answer_input: dict) -> dict:
    result = json.dumps(answer_input['result'], ensure_ascii=False)
    FILLED_ANSWER_PROMPT = ANSWER_PROMPT + '\n\n' + (
        f"question: {answer_input['question']}\n"
        f"sql: {answer_input['sql']}\n"
        f"result: {result}\n"
    )
    if answer_input.get('error'):
        FILLED_ANSWER_PROMPT += f"error: {answer_input['error']}\n"
    FILLED_ANSWER_PROMPT += f"image: {json.dumps(answer_input.get('image'))}\n"
    message = [
        {'role': 'system', 'content': 'You are an assistant; answer in the same language as the question.'},
        {'role': 'user', 'content': FILLED_ANSWER_PROMPT}
    ]
    response = openai.chat.completions.create(
        model="gpt-4.1-mini",
        messages = message,
        temperature = 0,
        max_tokens=300
    )
    answer_text = response.choices[0].message.content.strip()
    return {'query': answer_input['sql'], 'result': answer_input['result'], 'answer': answer_text, 'image': answer_input['image']}

# app = FastAPI()
# app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )

# class Question(BaseModel):
#     question: str
#     chart: bool | None = None

# class Answer(BaseModel):
#     sql: str
#     result: list
#     answer: str
#     image: str | None = None

# @app.post("/ask", response_model=Answer)
# def ask(q: Question):
#     need_chart = (q.chart is True) or (q.chart is None and detect_graph_request(q.question))
#     out = question_sql_answer(q.question, need_chart)
#     answer = Answer(sql=out['query'], result=out['result'], answer=out['answer'])
#     if(out['image'] != None):
#         answer.image = out['image']
#     return answer

if __name__ == "__main__":
    while True:
        question = input("Введіть ваше питання (або 'exit' для виходу): ")
        if question.lower() == 'exit':
            break
        need_chart = detect_graph_request(question)
        answer = question_sql_answer(question, need_chart)
        print("\nВідповідь:")
        print(answer['answer'])
        print("\nSQL Запит:")
        print(answer['query'])
        print("\nРезультат:")
        print(answer['result'])
        print("\nФото:")
        print(answer['image'])
        print("\n" + "="*50 + "\n")