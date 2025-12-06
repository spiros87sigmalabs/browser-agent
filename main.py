from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import asyncio

# ΣΩΣΤΑ IMPORTS από browser-use (από official README)
from browser_use import Agent, ChatBrowserUse, Browser

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
        # Native LLM από browser-use (δεν χρειάζεται key αν είναι default, αλλά βάζουμε για OpenAI)
        llm = ChatBrowserUse(
            model="gpt-4o-mini",
            openai_api_key=request.openai_api_key
        )

        # Browser instance (για καλύτερο control)
        browser = Browser()

        full_task = f"""
        WordPress URL: {request.wp_url}
        Username: {request.wp_user}
        Password: {request.wp_pass}

        ΕΡΓΑΣΙΑ:
        {request.task}

        Κάνε login στο /wp-admin και εκτέλεσε βήμα-βήμα.
        Στο τέλος γράψε τι έκανες.
        """

        agent = Agent(
            task=full_task,
            llm=llm,
            browser=browser,
            use_vision=True
        )

        result = await agent.run()

        return {"success": True, "result": str(result)}

    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/health")
def health():
    return {"status": "ok", "message": "LIVE & READY!"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("PORT", 8000)))
