from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from browser_use import Agent, ChatOpenAI
from browser_use.browser import BrowserProfile  # ← ΣΩΣΤΟ ΤΩΡΑ

app = FastAPI(title="Browser Agent Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class TaskRequest(BaseModel):
    task: str
    wp_url: str
    wp_user: str
    wp_pass: str
    openai_api_key: str

@app.post("/execute")
async def execute_task(request: TaskRequest):
    try:
        llm = ChatOpenAI(
            model="gpt-5-mini",
            api_key=request.openai_api_key
        )

        full_task = f"""
        WordPress URL: {request.wp_url}
        Username: {request.wp_user}
        Password: {request.wp_pass}

        ΕΡΓΑΣΙΑ: {request.task}

        ΒΗΜΑΤΑ:
        1. Πήγαινε στο {request.wp_url}/wp-admin
        2. Login με τα στοιχεία
        3. Κάνε την εργασία
        4. Γράψε τι έκανες
        """

        agent = Agent(
            task=full_task,
            llm=llm,
            use_vision=True,
            browser_profile=BrowserProfile(  # ← ΣΩΣΤΟ!
                headless=False,       # βλέπεις το Chrome
                slow_mo=1000,         # αργές κινήσεις
                timeout=45000,        # μεγάλο timeout
                wait_until="networkidle"
            )
        )

        result = await agent.run()
        return {"success": True, "result": str(result)}

    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/health")
def health():
    return {"status": "ok"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)