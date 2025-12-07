from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import json
import asyncio

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

async def stream_agent_logs(request: TaskRequest):
    """Generator που στέλνει real-time updates"""
    try:
        # Αρχικό μήνυμα
        yield f"data: {json.dumps({'type': 'info', 'message': '🚀 Εκκίνηση AI Agent...'})}\n\n"
        await asyncio.sleep(0.5)
        
        yield f"data: {json.dumps({'type': 'info', 'message': f'🌐 Σύνδεση στο {request.wp_url}'})}\n\n"
        await asyncio.sleep(0.5)
        
        # Δημιουργία LLM
        llm = ChatOpenAI(
            model="gpt-4o-mini",
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
        
        yield f"data: {json.dumps({'type': 'info', 'message': '🤖 Δημιουργία AI Agent...'})}\n\n"
        
        # Δημιουργία agent
        agent = Agent(
            task=full_task,
            llm=llm,
            use_vision=True,
            browser_profile=BrowserProfile(
                headless=True,
                slow_mo=500,
                timeout=60000,
                wait_until="networkidle",
                disable_security=False
            )
        )
        
        yield f"data: {json.dumps({'type': 'info', 'message': '🔥 Άνοιγμα Chrome browser...'})}\n\n"
        await asyncio.sleep(0.5)
        
        # Εκτέλεση με callback για real-time updates
        step_count = 0
        
        # Custom callback για κάθε action
        async def action_callback(action_info):
            nonlocal step_count
            step_count += 1
            
            # Παίρνουμε info από το action
            action_type = action_info.get('action', 'unknown')
            action_data = action_info.get('data', {})
            
            if action_type == 'click':
                msg = f"🖱️ Βήμα {step_count}: Κλικ σε '{action_data.get('element', 'στοιχείο')}'"
            elif action_type == 'type':
                msg = f"⌨️ Βήμα {step_count}: Γράφω '{action_data.get('text', '...')}'"
            elif action_type == 'navigate':
                msg = f"🧭 Βήμα {step_count}: Μετάβαση στο {action_data.get('url', 'νέα σελίδα')}"
            elif action_type == 'wait':
                msg = f"⏱️ Βήμα {step_count}: Αναμονή..."
            else:
                msg = f"⚡ Βήμα {step_count}: {action_type}"
            
            yield f"data: {json.dumps({'type': 'info', 'message': msg})}\n\n"
        
        # Εκτέλεση
        yield f"data: {json.dumps({'type': 'warning', 'message': '🧠 AI σκέφτεται...'})}\n\n"
        
        result = await agent.run()
        
        # Τελικό αποτέλεσμα
        yield f"data: {json.dumps({'type': 'success', 'message': '✅ Task ολοκληρώθηκε!'})}\n\n"
        await asyncio.sleep(0.5)
        
        # Parse result
        output = ""
        if hasattr(result, 'final_result'):
            output = str(result.final_result())
        elif hasattr(result, 'history') and result.history:
            output = "\n".join([str(h) for h in result.history[-5:]])
        else:
            output = str(result)
        
        # Στείλε το τελικό output
        for line in output.split('\n')[:10]:  # Πρώτες 10 γραμμές
            if line.strip():
                yield f"data: {json.dumps({'type': 'result', 'message': f'📄 {line}'})}\n\n"
                await asyncio.sleep(0.2)
        
        yield f"data: {json.dumps({'type': 'done', 'message': '🎉 Όλα έτοιμα!'})}\n\n"
        
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        yield f"data: {json.dumps({'type': 'error', 'message': f'❌ Σφάλμα: {error_msg}'})}\n\n"
        
        # Στείλε και το traceback
        tb = traceback.format_exc()
        for line in tb.split('\n')[:5]:
            if line.strip():
                yield f"data: {json.dumps({'type': 'error', 'message': line})}\n\n"

@app.post("/execute-stream")
async def execute_task_stream(request: TaskRequest):
    """Streaming endpoint για real-time updates"""
    return StreamingResponse(
        stream_agent_logs(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.post("/execute")
async def execute_task(request: TaskRequest):
    """Regular endpoint (για backward compatibility)"""
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
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

        agent = Agent(
            task=full_task,
            llm=llm,
            use_vision=True,
            browser_profile=BrowserProfile(
                headless=True,
                slow_mo=500,
                timeout=60000,
                wait_until="networkidle",
                disable_security=False
            )
        )

        result = await agent.run()
        
        output = ""
        if hasattr(result, 'final_result'):
            output = str(result.final_result())
        elif hasattr(result, 'history') and result.history:
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
        "version": "2.0.0 - Streaming Edition"
    }

@app.get("/")
def root():
    return {
        "name": "Browser Agent API",
        "endpoints": {
            "POST /execute": "Run task (regular)",
            "POST /execute-stream": "Run task (streaming)",
            "GET /health": "Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
