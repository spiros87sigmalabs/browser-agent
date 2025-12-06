from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

# ΣΩΣΤΟ IMPORT – από browser_use.llms
from browser_use import Agent
from browser_use.llms import ChatBrowserUse  # ΑΥΤΟ ΕΙΝΑΙ ΤΟ ΣΩΣΤΟ!

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
        # Χρησιμοποιούμε το NATIVE LLM της browser-use
        llm = ChatBrowserUse(
            model="gpt-4o-mini",
            openai_api_key=request.openai_api_key
        )

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
            use_vision=True,
            browser_profile={
                "headless": False,
                "slow_mo": 300,
                "timeout": 30000,
                "wait_until": "domcontentloaded"
            }
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
