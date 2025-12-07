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
import sys

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

# Railway Console Logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)

class TaskRequest(BaseModel):
    task: str
    wp_url: str
    wp_user: str
    wp_pass: str
    openai_api_key: str

def parse_log_line(line: str):
    """Parse and categorize log lines"""
    clean = re.sub(r'\x1b\[[0-9;]*m', '', line)
    
    if not clean.strip():
        return None, None, 0
    
    step = 0
    step_match = re.search(r'Step (\d+)', clean)
    if step_match:
        step = int(step_match.group(1))
    
    # Categorize by emoji/keyword
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
    elif '▶️' in clean or '⌨️' in clean or '🖱️' in clean or '🧭' in clean or '🔗' in clean:
        return 'action', clean.strip(), step
    elif '❌' in clean or 'ERROR' in clean or 'Failed' in clean or 'failed' in clean:
        return 'error', clean.strip(), step
    elif '✅' in clean or 'Success' in clean:
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
            
            # Print to Railway console for debugging
            print(f"[AGENT] {msg}", flush=True)
            
            log_type, message, step = parse_log_line(msg)
            if log_type and message:
                self.queue.put_nowait({
                    'type': log_type,
                    'message': message,
                    'step': step
                })
        except Exception as e:
            print(f"[ERROR] LogCapture: {e}", flush=True)

async def stream_agent_logs(request: TaskRequest):
    """Stream agent execution logs with optimized memory usage"""
    log_queue = asyncio.Queue()
    agent = None
    
    try:
        logger.info(f"🚀 New task: {request.task[:100]}")
        logger.info(f"🌐 Target: {request.wp_url}")
        
        # Setup browser_use logging
        for logger_name in ['browser_use']:
            log = logging.getLogger(logger_name)
            log.setLevel(logging.INFO)
            log.handlers.clear()
            
            handler = LogCapture(log_queue)
            handler.setFormatter(logging.Formatter('%(message)s'))
            log.addHandler(handler)
        
        yield f"data: {json.dumps({'type': 'system', 'message': '🚀 Initializing agent...', 'step': 0})}\n\n"
        
        # Initialize LLM
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=request.openai_api_key,
            temperature=0.1  # More deterministic
        )
        
        # Enhanced task prompt with explicit wait instructions
        full_task = f"""
WordPress Site: {request.wp_url}
Username: {request.wp_user}
Password: {request.wp_pass}

MAIN TASK: {request.task}

EXECUTION RULES (FOLLOW STRICTLY):
1. Navigate to {request.wp_url}/wp-admin
2. If redirected to login:
   - Enter username: {request.wp_user}
   - Enter password: {request.wp_pass}
   - Click "Log In" or "Σύνδεση" button
   - WAIT 8 seconds for dashboard to load completely
3. After successful login, verify you're on the dashboard
4. Execute the main task step by step
5. After EVERY action that changes the page (clicks, form submissions):
   - WAIT 5 seconds minimum
   - Verify the page loaded correctly
6. If an element click fails:
   - WAIT 5 seconds
   - Try scrolling to the element first
   - Try clicking again
7. When task is complete, provide a clear summary of what was accomplished

IMPORTANT NOTES:
- WordPress admin pages can be slow - always wait after navigation
- If you see "Node does not belong to document" error, the page changed - reload and try again
- Take your time - accuracy is more important than speed
"""
        
        logger.info("🔧 Creating browser agent with optimized settings...")
        
        # Create agent with production-ready settings
        agent = Agent(
            task=full_task,
            llm=llm,
            use_vision=False,  # Disabled for memory efficiency
            max_actions_per_step=5,  # Limit actions per step
            browser_profile=BrowserProfile(
                headless=True,
                slow_mo=500,  # Slow down for WordPress reliability
                timeout=150000,  # 150s generous timeout
                wait_until="networkidle",  # Wait for all network activity
                disable_security=True,
                extra_chromium_args=[
                    # Memory optimization
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    
                    # Performance
                    '--disable-extensions',
                    '--disable-background-networking',
                    '--disable-background-timer-throttling',
                    '--disable-backgrounding-occluded-windows',
                    '--disable-renderer-backgrounding',
                    '--disable-sync',
                    '--disable-translate',
                    '--disable-features=TranslateUI,BlinkGenPropertyTrees',
                    
                    # Resource limits
                    '--metrics-recording-only',
                    '--mute-audio',
                    '--no-first-run',
                    '--safebrowsing-disable-auto-update',
                    '--disable-blink-features=AutomationControlled',
                    
                    # Display
                    '--window-size=1920,1080',  # Larger viewport for better element detection
                    '--disable-infobars',
                    '--force-device-scale-factor=1',
                    
                    # Stability
                    '--disable-crash-reporter',
                    '--disable-in-process-stack-traces',
                    '--log-level=3',  # Minimal logging
                ]
            )
        )
        
        logger.info("✅ Agent ready - Starting execution...")
        yield f"data: {json.dumps({'type': 'system', 'message': '✅ Agent ready', 'step': 0})}\n\n"
        
        # Execute task
        async def run_task():
            try:
                return await agent.run()
            except Exception as e:
                logger.error(f"Task execution error: {e}")
                raise
        
        async def stream_logs():
            heartbeat_counter = 0
            while True:
                try:
                    log = await asyncio.wait_for(log_queue.get(), timeout=2.0)
                    yield log
                    heartbeat_counter = 0
                except asyncio.TimeoutError:
                    heartbeat_counter += 1
                    if heartbeat_counter % 3 == 0:
                        yield {'type': 'info', 'message': '💭 Agent is working...', 'step': 0}
                    await asyncio.sleep(2)
        
        # Run task and stream logs concurrently
        task = asyncio.create_task(run_task())
        log_gen = stream_logs()
        
        while not task.done():
            try:
                log_data = await asyncio.wait_for(log_gen.__anext__(), timeout=5.0)
                yield f"data: {json.dumps(log_data)}\n\n"
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'type': 'info', 'message': '⏳ Processing...', 'step': 0})}\n\n"
            except StopAsyncIteration:
                break
        
        # Get result
        result = await task
        
        logger.info("✅ Task completed successfully!")
        yield f"data: {json.dumps({'type': 'success', 'message': '✅ Task completed!', 'step': 999})}\n\n"
        
        # Stream result
        output = str(result) if result else "Task completed - no detailed output available"
        result_lines = output.split('\n')
        
        for i, line in enumerate(result_lines[:10]):  # First 10 lines
            if line.strip():
                logger.info(f"Result [{i+1}]: {line[:200]}")
                yield f"data: {json.dumps({'type': 'result', 'message': line[:250], 'step': 999})}\n\n"
        
        if len(result_lines) > 10:
            yield f"data: {json.dumps({'type': 'result', 'message': f'... and {len(result_lines)-10} more lines', 'step': 999})}\n\n"
        
        yield f"data: {json.dumps({'type': 'done', 'message': '🎉 All done!', 'step': 999})}\n\n"
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Fatal error: {error_msg}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'message': f'❌ Error: {error_msg[:200]}', 'step': 0})}\n\n"
    
    finally:
        # Cleanup
        if agent:
            try:
                logger.info("🧹 Closing browser...")
                await agent.browser.close()
                logger.info("✅ Browser closed")
            except Exception as e:
                logger.warning(f"Browser cleanup warning: {e}")
        
        # Force garbage collection
        gc.collect()
        logger.info("🧹 Memory cleanup complete")
        
        yield f"data: {json.dumps({'type': 'system', 'message': '🧹 Cleanup complete', 'step': 0})}\n\n"

@app.post("/execute-stream")
async def execute_task_stream(request: TaskRequest):
    """Stream task execution with real-time logs"""
    logger.info(f"📥 Stream request received: {request.task[:80]}...")
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
    """Execute task without streaming (simple response)"""
    agent = None
    try:
        logger.info(f"📥 Execute request: {request.task[:80]}...")
        
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            api_key=request.openai_api_key,
            temperature=0.1
        )

        full_task = f"""
WordPress: {request.wp_url}
Username: {request.wp_user}
Password: {request.wp_pass}

Task: {request.task}

Rules:
1. Navigate and login if needed
2. Wait 8 seconds after login
3. Execute task carefully
4. Wait 5 seconds after each page change
5. Report results
"""

        agent = Agent(
            task=full_task,
            llm=llm,
            use_vision=False,
            browser_profile=BrowserProfile(
                headless=True,
                slow_mo=500,
                timeout=150000,
                wait_until="networkidle",
                disable_security=True,
                extra_chromium_args=[
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-extensions',
                    '--window-size=1920,1080',
                ]
            )
        )

        result = await agent.run()
        output = str(result) if result else "Task completed successfully"
        
        logger.info("✅ Non-streaming task completed")
        return {"success": True, "result": output}

    except Exception as e:
        logger.error(f"❌ Error in non-streaming execution: {str(e)}", exc_info=True)
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
    """Health check endpoint"""
    return {"status": "ok", "service": "browser-agent", "version": "3.2.0"}

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "name": "Browser Agent API",
        "version": "3.2.0-production",
        "status": "operational",
        "endpoints": {
            "stream": "/execute-stream",
            "execute": "/execute",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting server on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
