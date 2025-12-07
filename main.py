from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os

from browser_use import Agent, ChatOpenAI
from browser_use.browser import BrowserProfile

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
        # ✅ ChatOpenAI από browser_use (όχι langchain!)
        llm = ChatOpenAI(
            model="gpt-4o-mini",  # ← ΣΩΣΤΟ model name
            api_key=request.openai_api_key
        )

        full_task = f"""
WordPress URL: {request.wp_url}
Username: {request.wp_user}
Password: {request.wp_pass}

ΕΡΓΑΣΙΑ: {request.task}

ΒΗΜΑΤΑ:
1. Πήγαινε στο {request.wp_url}/wp-admin
2. Συμπλήρωσε Username: {request.wp_user}
3. Συμπλήρωσε Password: {request.wp_pass}
4. Πάτα "Σύνδεση" ή "Log In"
5. Εκτέλεσε την εργασία βήμα-βήμα
6. Στο τέλος γράψε ΑΝΑΛΥΤΙΚΑ τι έκανες
"""

        # ✅ BrowserProfile για PRODUCTION (headless=True)
        agent = Agent(
            task=full_task,
            llm=llm,
            use_vision=True,
            browser_profile=BrowserProfile(
                headless=True,        # ← ΠΡΕΠΕΙ True για Render!
                slow_mo=500,          # λίγο πιο αργά
                timeout=60000,        # 60 sec timeout
                wait_until="networkidle",
                disable_security=False  # security ON
            )
        )

        result = await agent.run()
        
        # ✅ Better result parsing
        output = ""
        if hasattr(result, 'final_result'):
            output = str(result.final_result())
        elif hasattr(result, 'history') and result.history:
            # Πάρε τα τελευταία 5 messages
            output = "\n\n".join([str(h) for h in result.history[-5:]])
        else:
            output = str(result)
        
        return {
            "success": True, 
            "result": output,
            "model_used": "gpt-4o-mini"
        }

    except Exception as e:
        import traceback
        return {
            "success": False, 
            "error": f"{type(e).__name__}: {str(e)}",
            "traceback": traceback.format_exc()
        }

@app.get("/health")
def health():
    return {
        "status": "ok",
        "message": "Browser Agent LIVE!",
        "version": "1.0.0"
    }

@app.get("/")
def root():
    return {
        "name": "Browser Agent API",
        "endpoints": {
            "POST /execute": "Run browser automation task",
            "GET /health": "Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
