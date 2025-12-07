from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import json
import asyncio
import logging
import sys
from io import StringIO

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

class LogCapture(logging.Handler):
    """Custom handler που capture τα logs του browser_use"""
    def __init__(self, queue):
        super().__init__()
        self.queue = queue
        
    def emit(self, record):
        try:
            msg = self.format(record)
            self.queue.put_nowait({
                'level': record.levelname,
                'message': msg,
                'module': record.module
            })
        except:
            pass

async def stream_agent_logs(request: TaskRequest):
    """Generator με detailed streaming logs"""
    try:
        # Δημιουργία queue για logs
        log_queue = asyncio.Queue()
        
        # Setup custom logger για browser_use
        browser_logger = logging.getLogger('browser_use')
        browser_logger.setLevel(logging.INFO)
        
        # Προσθήκη custom handler
        handler = LogCapture(log_queue)
        handler.setFormatter(logging.Formatter('%(message)s'))
        browser_logger.addHandler(handler)
        
        # Αρχικά μηνύματα
        yield f"data: {json.dumps({'type': 'info', 'message': '🚀 Εκκίνηση AI Agent Pro...', 'step': 0})}\n\n"
        await asyncio.sleep(0.3)
        
        yield f"data: {json.dumps({'type': 'info', 'message': f'🌐 Target: {request.wp_url}', 'step': 0})}\n\n"
        await asyncio.sleep(0.3)
        
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
5. Εκτέλεσε την εργασία βήμα-βήμα με προσοχή
6. Στο τέλος γράψε ΑΝΑΛΥΤΙΚΑ τι έκανες
"""
        
        yield f"data: {json.dumps({'type': 'system', 'message': '🤖 Initializing AI Brain...', 'step': 0})}\n\n"
        
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
        
        yield f"data: {json.dumps({'type': 'system', 'message': '🔥 Launching Chrome Browser...', 'step': 0})}\n\n"
        await asyncio.sleep(0.3)
        
        # Task για εκτέλεση agent
        async def run_agent():
            return await agent.run()
        
        # Task για monitoring logs
        async def monitor_logs():
            step_counter = 0
            while True:
                try:
                    log_entry = await asyncio.wait_for(log_queue.get(), timeout=0.1)
                    
                    msg = log_entry['message']
                    
                    # Parse διαφορετικά types
                    if '📍 Step' in msg:
                        step_counter += 1
                        step_num = msg.split('Step')[1].split(':')[0].strip()
                        yield f"data: {json.dumps({'type': 'step', 'message': f'📍 Step {step_num}', 'step': step_counter})}\n\n"
                    
                    elif '👍 Eval:' in msg:
                        eval_text = msg.split('Eval:')[1].strip()
                        yield f"data: {json.dumps({'type': 'eval', 'message': f'👍 {eval_text}', 'step': step_counter})}\n\n"
                    
                    elif '🧠 Memory:' in msg:
                        memory_text = msg.split('Memory:')[1].strip()
                        yield f"data: {json.dumps({'type': 'memory', 'message': f'🧠 {memory_text}', 'step': step_counter})}\n\n"
                    
                    elif '🎯 Next goal:' in msg:
                        goal_text = msg.split('Next goal:')[1].strip()
                        yield f"data: {json.dumps({'type': 'goal', 'message': f'🎯 {goal_text}', 'step': step_counter})}\n\n"
                    
                    elif '▶️' in msg:
                        action_text = msg.split('▶️')[1].strip()
                        yield f"data: {json.dumps({'type': 'action', 'message': f'▶️ {action_text}', 'step': step_counter})}\n\n"
                    
                    elif '🖱️' in msg or 'click' in msg.lower():
                        yield f"data: {json.dumps({'type': 'action', 'message': f'🖱️ {msg}', 'step': step_counter})}\n\n"
                    
                    elif '⌨️' in msg or 'type' in msg.lower():
                        yield f"data: {json.dumps({'type': 'action', 'message': f'⌨️ {msg}', 'step': step_counter})}\n\n"
                    
                    elif '🧭' in msg or 'navigate' in msg.lower():
                        yield f"data: {json.dumps({'type': 'action', 'message': f'🧭 {msg}', 'step': step_counter})}\n\n"
                    
                    elif 'ERROR' in log_entry['level']:
                        yield f"data: {json.dumps({'type': 'error', 'message': f'❌ {msg}', 'step': step_counter})}\n\n"
                    
                    else:
                        # Generic info
                        yield f"data: {json.dumps({'type': 'info', 'message': msg, 'step': step_counter})}\n\n"
                    
                    await asyncio.sleep(0.05)
                    
                except asyncio.TimeoutError:
                    await asyncio.sleep(0.1)
                    continue
                except Exception as e:
                    break
        
        # Εκτέλεση παράλληλα
        agent_task = asyncio.create_task(run_agent())
        
        # Stream logs
        async for log_data in monitor_logs():
            yield log_data
            
            # Check αν τελείωσε το agent
            if agent_task.done():
                break
        
        # Περίμενε να τελειώσει
        result = await agent_task
        
        # Cleanup
        browser_logger.removeHandler(handler)
        
        # Τελικό αποτέλεσμα
        yield f"data: {json.dumps({'type': 'success', 'message': '✅ Task Completed Successfully!', 'step': 999})}\n\n"
        await asyncio.sleep(0.3)
        
        # Parse result
        output = ""
        if hasattr(result, 'final_result'):
            output = str(result.final_result())
        elif hasattr(result, 'history') and result.history:
            output = "\n".join([str(h) for h in result.history[-5:]])
        else:
            output = str(result)
        
        # Στείλε summary
        yield f"data: {json.dumps({'type': 'result', 'message': '📋 SUMMARY', 'step': 999})}\n\n"
        
        for line in output.split('\n')[:15]:
            if line.strip():
                yield f"data: {json.dumps({'type': 'result', 'message': line.strip(), 'step': 999})}\n\n"
                await asyncio.sleep(0.1)
        
        yield f"data: {json.dumps({'type': 'done', 'message': '🎉 All Done!', 'step': 999})}\n\n"
        
    except Exception as e:
        import traceback
        error_msg = f"{type(e).__name__}: {str(e)}"
        yield f"data: {json.dumps({'type': 'error', 'message': f'❌ Fatal Error: {error_msg}', 'step': 0})}\n\n"
        
        tb = traceback.format_exc()
        for line in tb.split('\n')[:8]:
            if line.strip():
                yield f"data: {json.dumps({'type': 'error', 'message': line, 'step': 0})}\n\n"

@app.post("/execute-stream")
async def execute_task_stream(request: TaskRequest):
    """Streaming endpoint με detailed logs"""
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
        "message": "Browser Agent Pro LIVE!",
        "version": "3.0.0 - Enhanced Logging"
    }

@app.get("/")
def root():
    return {
        "name": "Browser Agent Pro API",
        "endpoints": {
            "POST /execute": "Run task (regular)",
            "POST /execute-stream": "Run task (streaming with detailed logs)",
            "GET /health": "Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)

