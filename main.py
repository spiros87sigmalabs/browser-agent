from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
import os
import json
import asyncio
import logging
import re
import gc

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

def parse_log_line(line: str):
    """Parse log line"""
    clean = re.sub(r'\x1b\[[0-9;]*m', '', line)
    
    if not clean.strip():
        return None, None, 0
    
    step = 0
    step_match = re.search(r'Step (\d+)', clean)
    if step_match:
        step = int(step_match.group(1))
    
    if '📍 Step' in clean:
        return 'step', clean.strip(), step
    elif '👍 Eval:' in clean and ('Success' in clean or 'successfully' in clean):
        return 'eval', clean.strip(), step
    elif '⚠️ Eval:' in clean or 'Failure' in clean:
        return 'warning', clean.strip(), step
    elif '🧠' in clean and '[Agent]' in clean:
        return 'memory', clean.strip(), step
    elif '🎯' in clean and 'Task:' not in clean:
        return 'goal', clean.strip(), step
    elif '▶️' in clean or '⌨️' in clean or '🖱️' in clean or '🧭' in clean:
        return 'action', clean.strip(), step
    elif '❌' in clean or 'ERROR' in clean or 'Failed' in clean:
        return 'error', clean.strip(), step
    elif '✅' in clean:
        return 'success', clean.strip(), step
    elif 'Starting' in clean or 'Downloading' in clean or 'viewport' in clean:
        return 'system', clean.strip(), step
    else:
        return 'info', clean.strip(), step

class LogCapture(logging.Handler):
    def __init__(self, queue):
        super().__init__()
        self.queue = queue
        
    def emit(self, record):
        try:
            msg = self.format(record)
            log_type, message, step = parse_log_line(msg)
            if log_type and message:
                self.queue.put_nowait({
                    'type': log_type,
                    'message': message,
                    'step': step
                })
        except:
            pass

async def stream_agent_logs(request: TaskRequest):
    """Streaming με memory optimization"""
    log_queue = asyncio.Queue()
    agent = None
    
    try:
        # Setup logging
        for logger_name in ['browser_use']:
            logger = logging.getLogger(logger_name)
            logger.setLevel(logging.INFO)
            logger.handlers.clear()
            
            handler = LogCapture(log_queue)
            handler.setFormatter(logging.Formatter('%(message)s'))
            logger.addHandler(handler)
        
        yield f"data: {json.dumps({'type': 'system', 'message': '🚀 Starting...', 'step': 0})}\n\n"
        
        # LLM
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=request.openai_api_key
        )
        
        full_task = f"""
WordPress: {request.wp_url}
User: {request.wp_user}
Pass: {request.wp_pass}

TASK: {request.task}

STEPS:
1. Go to {request.wp_url}/wp-admin
2. Login if needed
3. Complete task
4. Report what you did
"""
        
        # MEMORY-OPTIMIZED BROWSER
        agent = Agent(
            task=full_task,
            llm=llm,
            use_vision=False,  # ❌ NO VISION
            browser_profile=BrowserProfile(
                headless=True,
                slow_mo=100,  # Faster
                timeout=90000,  # 90s
                wait_until="domcontentloaded",
                disable_security=True,
                # 🔥 MEMORY OPTIMIZATION
                extra_chromium_args=[
                    '--disable-dev-shm-usage',  # Μην χρησιμοποιείς /dev/shm
                    '--disable-gpu',  # No GPU
                    '--no-sandbox',  # No sandbox
                    '--disable-setuid-sandbox',
                    '--disable-extensions',  # No extensions
                    '--disable-background-networking',
                    '--disable-background-timer-throttling',
                    '--disable-backgrounding-occluded-windows',
                    '--disable-renderer-backgrounding',
                    '--disable-sync',
                    '--disable-translate',
                    '--disable-features=TranslateUI',
                    '--metrics-recording-only',
                    '--mute-audio',
                    '--no-first-run',
                    '--safebrowsing-disable-auto-update',
                    '--disable-blink-features=AutomationControlled',
                    '--window-size=1280,720',  # Μικρότερο window
                    '--disable-infobars',
                ]
            )
        )
        
        yield f"data: {json.dumps({'type': 'system', 'message': '🔥 Chrome ready', 'step': 0})}\n\n"
        
        # Run agent
        async def run_task():
            return await agent.run()
        
        async def stream_logs():
            while True:
                try:
                    log = await asyncio.wait_for(log_queue.get(), timeout=1.0)
                    yield log
                except asyncio.TimeoutError:
                    yield {'type': 'info', 'message': '💭 Working...', 'step': 0}
                    await asyncio.sleep(2)
        
        task = asyncio.create_task(run_task())
        log_gen = stream_logs()
        
        while not task.done():
            try:
                log_data = await asyncio.wait_for(log_gen.__anext__(), timeout=3.0)
                yield f"data: {json.dumps(log_data)}\n\n"
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'type': 'info', 'message': '⏳ Processing...', 'step': 0})}\n\n"
            except StopAsyncIteration:
                break
        
        result = await task
        
        yield f"data: {json.dumps({'type': 'success', 'message': '✅ Done!', 'step': 999})}\n\n"
        
        # Result
        output = str(result) if result else "No output"
        for line in output.split('\n')[:8]:
            if line.strip():
                yield f"data: {json.dumps({'type': 'result', 'message': line[:200], 'step': 999})}\n\n"
        
        yield f"data: {json.dumps({'type': 'done', 'message': '🎉 Complete', 'step': 999})}\n\n"
        
    except Exception as e:
        yield f"data: {json.dumps({'type': 'error', 'message': f'❌ {str(e)}', 'step': 0})}\n\n"
    
    finally:
        # 🔥 CLEANUP MEMORY
        if agent:
            try:
                await agent.browser.close()
            except:
                pass
        
        # Force garbage collection
        gc.collect()
        
        yield f"data: {json.dumps({'type': 'system', 'message': '🧹 Cleanup done', 'step': 0})}\n\n"

@app.post("/execute-stream")
async def execute_task_stream(request: TaskRequest):
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
    agent = None
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=request.openai_api_key
        )

        full_task = f"""
WordPress: {request.wp_url}
User: {request.wp_user}
Pass: {request.wp_pass}
Task: {request.task}
"""

        agent = Agent(
            task=full_task,
            llm=llm,
            use_vision=False,
            browser_profile=BrowserProfile(
                headless=True,
                slow_mo=100,
                timeout=90000,
                wait_until="domcontentloaded",
                disable_security=True,
                extra_chromium_args=[
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-extensions',
                ]
            )
        )

        result = await agent.run()
        output = str(result) if result else "Done"
        
        return {"success": True, "result": output}

    except Exception as e:
        return {"success": False, "error": str(e)}
    
    finally:
        if agent:
            try:
                await agent.browser.close()
            except:
                pass
        gc.collect()

@app.get("/health")
def health():
    return {"status": "ok"}

@app.get("/")
def root():
    return {"name": "Browser Agent API", "version": "3.0.0-optimized"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
