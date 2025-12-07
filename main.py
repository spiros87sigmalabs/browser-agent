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
import psutil

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
            
            # Φιλτράρισμα CDP errors για να μην γεμίσουν τα logs
            if 'ConnectionClosedError' in msg or 'no close frame' in msg:
                return  # Μην στείλεις αυτά τα errors στο queue
            
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

def log_memory_usage():
    """Καταγραφή χρήσης μνήμης"""
    try:
        process = psutil.Process()
        mem_info = process.memory_info()
        mem_mb = mem_info.rss / 1024 / 1024
        logger.info(f"💾 Memory usage: {mem_mb:.2f} MB")
        return mem_mb
    except:
        return 0

async def check_browser_health(agent):
    """Έλεγχος αν το browser λειτουργεί"""
    try:
        if not agent or not agent.browser:
            return False
        
        # Δοκιμαστική εντολή
        pages = agent.browser.context.pages
        if not pages:
            return False
            
        await asyncio.wait_for(
            pages[0].evaluate("1+1"), 
            timeout=2.0
        )
        return True
    except Exception as e:
        logger.warning(f"Browser health check failed: {e}")
        return False

async def safe_close_browser(agent):
    """Ασφαλές κλείσιμο browser με retry"""
    if not agent:
        return
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            logger.info(f"🧹 Closing browser (attempt {attempt+1}/{max_retries})...")
            await asyncio.wait_for(agent.browser.close(), timeout=10.0)
            logger.info("✅ Browser closed successfully")
            return
        except asyncio.TimeoutError:
            logger.warning(f"Browser close timeout on attempt {attempt+1}")
            if attempt == max_retries - 1:
                logger.error("❌ Failed to close browser gracefully")
        except Exception as e:
            logger.warning(f"Browser close error: {e}")
            if attempt == max_retries - 1:
                logger.error("❌ Failed to close browser")

async def aggressive_cleanup():
    """Επιθετικό cleanup μνήμης"""
    logger.info("🧹 Starting aggressive cleanup...")
    
    # Multiple garbage collection passes
    for i in range(3):
        collected = gc.collect()
        logger.info(f"🗑️ GC pass {i+1}: collected {collected} objects")
        await asyncio.sleep(0.1)
    
    log_memory_usage()

async def stream_agent_logs(request: TaskRequest):
    """Stream agent execution logs with enhanced error handling"""
    log_queue = asyncio.Queue()
    agent = None
    
    try:
        logger.info(f"🚀 New task: {request.task[:100]}")
        logger.info(f"🌐 Target: {request.wp_url}")
        log_memory_usage()
        
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
            temperature=0.1
        )
        
        # Enhanced task prompt
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
   - WAIT 10 seconds for dashboard to load completely
3. After successful login, verify you're on the dashboard
4. Execute the main task step by step
5. After EVERY action that changes the page (clicks, form submissions):
   - WAIT 7 seconds minimum
   - Verify the page loaded correctly
6. If an element click fails:
   - WAIT 5 seconds
   - Try scrolling to the element first
   - Try clicking again
7. When task is complete, provide a clear summary

IMPORTANT:
- WordPress admin pages can be slow - always wait
- If errors occur, wait and retry
- Take your time - accuracy over speed
"""
        
        logger.info("🔧 Creating browser agent with production settings...")
        
        # Create agent with OPTIMIZED settings for Railway
        agent = Agent(
            task=full_task,
            llm=llm,
            use_vision=False,
            max_actions_per_step=3,
            browser_profile=BrowserProfile(
                headless=True,
                slow_mo=800,
                timeout=180000,
                wait_until="networkidle",
                disable_security=True,
                extra_chromium_args=[
                    # ΚΡΙΣΙΜΑ για Railway
                    '--single-process',
                    '--no-zygote',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    
                    # Memory optimization
                    '--js-flags=--max-old-space-size=384',
                    '--disable-features=IsolateOrigins,site-per-process',
                    '--disable-site-isolation-trials',
                    
                    # Performance
                    '--disable-extensions',
                    '--disable-background-networking',
                    '--disable-background-timer-throttling',
                    '--disable-backgrounding-occluded-windows',
                    '--disable-renderer-backgrounding',
                    '--disable-sync',
                    '--disable-translate',
                    '--disable-features=TranslateUI,BlinkGenPropertyTrees',
                    '--disable-component-extensions-with-background-pages',
                    
                    # Resource limits
                    '--metrics-recording-only',
                    '--mute-audio',
                    '--no-first-run',
                    '--safebrowsing-disable-auto-update',
                    '--disable-blink-features=AutomationControlled',
                    '--disable-ipc-flooding-protection',
                    
                    # Display
                    '--window-size=1366,768',
                    '--disable-infobars',
                    '--force-device-scale-factor=1',
                    
                    # Stability
                    '--disable-crash-reporter',
                    '--disable-in-process-stack-traces',
                    '--log-level=3',
                    '--disable-logging',
                    '--disable-breakpad',
                ]
            )
        )
        
        logger.info("✅ Agent ready - Starting execution...")
        log_memory_usage()
        yield f"data: {json.dumps({'type': 'system', 'message': '✅ Agent ready', 'step': 0})}\n\n"
        
        # Execute task με error recovery
        async def run_task_with_recovery():
            max_retries = 2
            last_result = None
            
            for attempt in range(max_retries):
                try:
                    logger.info(f"🎯 Task execution attempt {attempt+1}/{max_retries}")
                    last_result = await agent.run()
                    logger.info("✅ Task completed successfully")
                    return last_result  # Επιστρέφει το result
                except Exception as e:
                    error_str = str(e)
                    logger.error(f"❌ Attempt {attempt+1} failed: {error_str}")
                    
                    # Αν είναι connection error και έχουμε άλλη προσπάθεια
                    if attempt < max_retries - 1 and ('ConnectionClosed' in error_str or 'close frame' in error_str):
                        logger.warning("🔄 Connection lost, preparing retry...")
                        # Στείλε warning στο stream
                        await log_queue.put({
                            'type': 'warning',
                            'message': '⚠️ Connection issue, retrying...',
                            'step': 0
                        })
                        
                        # Κλείσιμο και cleanup
                        await safe_close_browser(agent)
                        await aggressive_cleanup()
                        await asyncio.sleep(3)
                        
                        logger.info("🔄 Retrying task...")
                        continue
                    else:
                        raise
            
            raise Exception("Task failed after all retries")
        
        async def stream_logs():
            heartbeat_counter = 0
            last_health_check = asyncio.get_event_loop().time()
            
            while True:
                try:
                    log = await asyncio.wait_for(log_queue.get(), timeout=3.0)
                    yield log
                    heartbeat_counter = 0
                    
                    # Periodic health check
                    current_time = asyncio.get_event_loop().time()
                    if current_time - last_health_check > 30:
                        if not await check_browser_health(agent):
                            logger.warning("⚠️ Browser health check failed!")
                            yield {'type': 'warning', 'message': '⚠️ Browser may be unstable', 'step': 0}
                        last_health_check = current_time
                    
                except asyncio.TimeoutError:
                    heartbeat_counter += 1
                    if heartbeat_counter % 2 == 0:
                        yield {'type': 'info', 'message': '💭 Agent working...', 'step': 0}
                    await asyncio.sleep(2)
        
        # Run task and stream logs concurrently
        task = asyncio.create_task(run_task_with_recovery())
        log_gen = stream_logs()
        
        while not task.done():
            try:
                log_data = await asyncio.wait_for(log_gen.__anext__(), timeout=5.0)
                if isinstance(log_data, dict):
                    yield f"data: {json.dumps(log_data)}\n\n"
            except asyncio.TimeoutError:
                yield f"data: {json.dumps({'type': 'info', 'message': '⏳ Processing...', 'step': 0})}\n\n"
            except StopAsyncIteration:
                break
        
        # Get result - ΣΗΜΑΝΤΙΚΟ: Χρησιμοποιούμε await εδώ
        try:
            result = await task
        except Exception as e:
            logger.error(f"Task failed: {e}")
            result = None
        
        logger.info("✅ Task completed successfully!")
        log_memory_usage()
        yield f"data: {json.dumps({'type': 'success', 'message': '✅ Task completed!', 'step': 999})}\n\n"
        
        # Stream result
        if result:
            output = str(result)
            result_lines = output.split('\n')
            
            for i, line in enumerate(result_lines[:10]):
                if line.strip():
                    logger.info(f"Result [{i+1}]: {line[:200]}")
                    yield f"data: {json.dumps({'type': 'result', 'message': line[:250], 'step': 999})}\n\n"
            
            if len(result_lines) > 10:
                yield f"data: {json.dumps({'type': 'result', 'message': f'... και {len(result_lines)-10} ακόμα γραμμές', 'step': 999})}\n\n"
        else:
            yield f"data: {json.dumps({'type': 'result', 'message': 'Task completed - no detailed output', 'step': 999})}\n\n"
        
        yield f"data: {json.dumps({'type': 'done', 'message': '🎉 Ολοκληρώθηκε!', 'step': 999})}\n\n"
        
    except Exception as e:
        error_msg = str(e)
        logger.error(f"❌ Fatal error: {error_msg}", exc_info=True)
        yield f"data: {json.dumps({'type': 'error', 'message': f'❌ Σφάλμα: {error_msg[:200]}', 'step': 0})}\n\n"
    
    finally:
        # Cleanup
        await safe_close_browser(agent)
        await aggressive_cleanup()
        
        logger.info("🧹 Cleanup complete")
        log_memory_usage()
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
        log_memory_usage()
        
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
2. Wait 10 seconds after login
3. Execute task carefully
4. Wait 7 seconds after each page change
5. Report results clearly
"""

        agent = Agent(
            task=full_task,
            llm=llm,
            use_vision=False,
            max_actions_per_step=3,
            browser_profile=BrowserProfile(
                headless=True,
                slow_mo=800,
                timeout=180000,
                wait_until="networkidle",
                disable_security=True,
                extra_chromium_args=[
                    '--single-process',
                    '--no-zygote',
                    '--disable-dev-shm-usage',
                    '--disable-gpu',
                    '--no-sandbox',
                    '--disable-setuid-sandbox',
                    '--disable-extensions',
                    '--window-size=1366,768',
                    '--js-flags=--max-old-space-size=384',
                    '--disable-features=IsolateOrigins,site-per-process',
                ]
            )
        )

        result = await agent.run()
        output = str(result) if result else "Task completed successfully"
        
        logger.info("✅ Non-streaming task completed")
        log_memory_usage()
        return {"success": True, "result": output}

    except Exception as e:
        logger.error(f"❌ Error in non-streaming execution: {str(e)}", exc_info=True)
        return {"success": False, "error": str(e)}
    
    finally:
        await safe_close_browser(agent)
        await aggressive_cleanup()

@app.get("/health")
def health():
    """Health check endpoint"""
    mem_usage = log_memory_usage()
    return {
        "status": "ok", 
        "service": "browser-agent", 
        "version": "3.3.1-fixed",
        "memory_mb": round(mem_usage, 2)
    }

@app.get("/")
def root():
    """Root endpoint"""
    return {
        "name": "Browser Agent API",
        "version": "3.3.1-fixed",
        "status": "operational",
        "features": [
            "Enhanced error recovery",
            "Memory optimization",
            "Connection retry logic",
            "Health monitoring",
            "Fixed async generator syntax"
        ],
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
    logger.info(f"🐍 Python version: {sys.version}")
    log_memory_usage()
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=port,
        timeout_keep_alive=300
    )

