from fastapi import FastAPI, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from playwright.async_api import async_playwright
import asyncio
import json
import logging
import sys
from typing import Optional
import uuid

app = FastAPI(title="WordPress Automation - No AI")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True
)
logger = logging.getLogger(__name__)

class TaskRequest(BaseModel):
    task_type: str  # 'yoast_seo', 'viva_wallet', 'clear_cache', 'custom'
    wp_url: str
    wp_user: str
    wp_pass: str
    custom_actions: Optional[list] = None  # Για custom tasks

# In-memory task storage (σε production βάλε Redis/DB)
tasks = {}

async def send_log(queue: asyncio.Queue, log_type: str, message: str, step: int = 0):
    """Στείλε log message"""
    await queue.put({
        'type': log_type,
        'message': message,
        'step': step
    })
    logger.info(f"[{log_type.upper()}] {message}")

async def wp_login(page, wp_url: str, wp_user: str, wp_pass: str, log_queue: asyncio.Queue):
    """Login στο WordPress"""
    await send_log(log_queue, 'navigate', f'🧭 Πηγαίνω στο {wp_url}/wp-admin', 1)
    
    await page.goto(f'{wp_url}/wp-admin', wait_until='networkidle', timeout=60000)
    await asyncio.sleep(2)
    
    # Έλεγξε αν είναι ήδη logged in
    if '/wp-admin/index.php' in page.url or 'dashboard' in page.url.lower():
        await send_log(log_queue, 'success', '✅ Ήδη συνδεδεμένος!', 1)
        return True
    
    await send_log(log_queue, 'type', f'⌨️ Γράφω username: {wp_user}', 1)
    await page.fill('input[name="log"]', wp_user)
    
    await send_log(log_queue, 'type', '⌨️ Γράφω password', 1)
    await page.fill('input[name="pwd"]', wp_pass)
    
    await send_log(log_queue, 'click', '🖱️ Πατάω Login', 1)
    await page.click('input[type="submit"]')
    
    await send_log(log_queue, 'wait', '⏳ Περιμένω login...', 1)
    await page.wait_for_load_state('networkidle', timeout=60000)
    await asyncio.sleep(3)
    
    await send_log(log_queue, 'success', '✅ Συνδέθηκα επιτυχώς!', 1)
    return True

async def task_yoast_seo(page, wp_url: str, log_queue: asyncio.Queue):
    """Yoast SEO Optimization"""
    await send_log(log_queue, 'step', '📍 Step 2: Yoast SEO για όλα τα προϊόντα', 2)
    
    # Πήγαινε στα Products
    await send_log(log_queue, 'navigate', '🧭 Πηγαίνω στα Products', 2)
    await page.goto(f'{wp_url}/wp-admin/edit.php?post_type=product', wait_until='networkidle')
    await asyncio.sleep(2)
    
    # Πάρε όλα τα product links
    product_links = await page.locator('a.row-title').all()
    product_count = len(product_links)
    
    await send_log(log_queue, 'success', f'✅ Βρήκα {product_count} προϊόντα', 2)
    
    for i in range(min(product_count, 5)):  # Limit στα 5 πρώτα για demo
        await send_log(log_queue, 'step', f'📍 Step {3+i}: Επεξεργασία προϊόντος {i+1}/{product_count}', 3+i)
        
        # Ξανά-fetch τα links (η σελίδα ανανεώνεται)
        await page.goto(f'{wp_url}/wp-admin/edit.php?post_type=product', wait_until='networkidle')
        await asyncio.sleep(1)
        
        product_links = await page.locator('a.row-title').all()
        if i >= len(product_links):
            break
        
        product_name = await product_links[i].text_content()
        await send_log(log_queue, 'click', f'🖱️ Ανοίγω: {product_name}', 3+i)
        
        await product_links[i].click()
        await page.wait_for_load_state('networkidle')
        await asyncio.sleep(2)
        
        # Scroll στο Yoast section
        try:
            await send_log(log_queue, 'navigate', '🔍 Ψάχνω Yoast SEO section...', 3+i)
            
            # Yoast focus keyword
            focus_keyword_input = page.locator('input[name="yoast_wpseo_focuskw"]').first
            if await focus_keyword_input.count() > 0:
                await send_log(log_queue, 'type', f'⌨️ Focus keyword: {product_name}', 3+i)
                await focus_keyword_input.fill(product_name)
            
            # SEO Title
            seo_title_input = page.locator('input[name="yoast_wpseo_title"]').first
            if await seo_title_input.count() > 0:
                await send_log(log_queue, 'type', f'⌨️ SEO Title: {product_name}', 3+i)
                await seo_title_input.fill(f'{product_name} - Shop Now')
            
            # Meta Description
            meta_desc_input = page.locator('textarea[name="yoast_wpseo_metadesc"]').first
            if await meta_desc_input.count() > 0:
                await send_log(log_queue, 'type', '⌨️ Meta Description', 3+i)
                await meta_desc_input.fill(f'Buy {product_name} online. Best prices and quality.')
            
            # Update
            await send_log(log_queue, 'click', '🖱️ Πατάω Update', 3+i)
            await page.click('button:has-text("Update"), button:has-text("Ενημέρωση")')
            await asyncio.sleep(2)
            
            await send_log(log_queue, 'success', f'✅ Ενημερώθηκε: {product_name}', 3+i)
        
        except Exception as e:
            await send_log(log_queue, 'error', f'⚠️ Σφάλμα σε {product_name}: {str(e)}', 3+i)
    
    await send_log(log_queue, 'success', f'✅ Ολοκληρώθηκε Yoast SEO για {min(product_count, 5)} προϊόντα!', 99)

async def task_viva_wallet(page, wp_url: str, log_queue: asyncio.Queue):
    """Install & Activate Viva Wallet"""
    await send_log(log_queue, 'step', '📍 Step 2: Εγκατάσταση Viva Wallet', 2)
    
    # Πήγαινε στα Plugins
    await send_log(log_queue, 'navigate', '🧭 Πηγαίνω στα Plugins', 2)
    await page.goto(f'{wp_url}/wp-admin/plugins.php', wait_until='networkidle')
    await asyncio.sleep(2)
    
    # Ψάξε αν υπάρχει ήδη
    page_content = await page.content()
    
    if 'viva' in page_content.lower() or 'viva-wallet' in page_content.lower():
        await send_log(log_queue, 'success', '✅ Viva Wallet ήδη εγκατεστημένο', 2)
        
        # Τσέκαρε αν χρειάζεται activation
        activate_link = page.locator('tr:has-text("Viva") .activate a, tr:has-text("viva") .activate a').first
        
        if await activate_link.count() > 0:
            await send_log(log_queue, 'click', '🖱️ Ενεργοποιώ το plugin', 2)
            await activate_link.click()
            await page.wait_for_load_state('networkidle')
            await asyncio.sleep(2)
            await send_log(log_queue, 'success', '✅ Plugin ενεργοποιήθηκε!', 2)
        else:
            await send_log(log_queue, 'success', '✅ Plugin ήδη ενεργό', 2)
    else:
        # Εγκατάσταση
        await send_log(log_queue, 'step', '📍 Step 3: Κατέβασμα Viva Wallet', 3)
        await send_log(log_queue, 'click', '🖱️ Πατάω Add New', 3)
        
        await page.click('a:has-text("Add New"), a:has-text("Προσθήκη")')
        await page.wait_for_load_state('networkidle')
        await asyncio.sleep(2)
        
        await send_log(log_queue, 'type', '⌨️ Ψάχνω "Viva Wallet"', 3)
        await page.fill('input[name="s"]', 'Viva Wallet for WooCommerce')
        await page.press('input[name="s"]', 'Enter')
        await asyncio.sleep(3)
        
        # Install
        install_button = page.locator('a:has-text("Install Now")').first
        if await install_button.count() > 0:
            await send_log(log_queue, 'click', '🖱️ Πατάω Install Now', 3)
            await install_button.click()
            
            await send_log(log_queue, 'wait', '⏳ Περιμένω εγκατάσταση (10s)...', 3)
            await asyncio.sleep(10)
            
            # Activate
            activate_button = page.locator('a:has-text("Activate")').first
            if await activate_button.count() > 0:
                await send_log(log_queue, 'click', '🖱️ Πατάω Activate', 3)
                await activate_button.click()
                await asyncio.sleep(3)
                await send_log(log_queue, 'success', '✅ Plugin εγκαταστάθηκε και ενεργοποιήθηκε!', 3)
            else:
                await send_log(log_queue, 'error', '⚠️ Δεν βρήκα Activate button', 3)
        else:
            await send_log(log_queue, 'error', '⚠️ Δεν βρήκα Install button', 3)
    
    # Enable στο WooCommerce
    await send_log(log_queue, 'step', '📍 Step 4: Ενεργοποίηση στο WooCommerce', 4)
    await send_log(log_queue, 'navigate', '🧭 Πηγαίνω στα Payments', 4)
    
    await page.goto(f'{wp_url}/wp-admin/admin.php?page=wc-settings&tab=checkout', wait_until='networkidle')
    await asyncio.sleep(2)
    
    # Βρες το Viva Wallet toggle
    viva_toggle = page.locator('tr:has-text("Viva") .woocommerce-input-toggle, tr:has-text("viva") .woocommerce-input-toggle').first
    
    if await viva_toggle.count() > 0:
        is_enabled = await viva_toggle.get_attribute('aria-checked')
        
        if is_enabled == 'false':
            await send_log(log_queue, 'click', '🖱️ Ενεργοποιώ Viva Wallet payment', 4)
            await viva_toggle.click()
            await asyncio.sleep(1)
        
        await send_log(log_queue, 'success', '✅ Viva Wallet ενεργοποιημένο στα payments!', 4)
    else:
        await send_log(log_queue, 'error', '⚠️ Δεν βρήκα Viva Wallet στα payments', 4)
    
    await send_log(log_queue, 'success', '🎉 Ολοκληρώθηκε η εγκατάσταση Viva Wallet!', 99)

async def task_clear_cache(page, wp_url: str, log_queue: asyncio.Queue):
    """Clear WordPress Cache"""
    await send_log(log_queue, 'step', '📍 Step 2: Καθαρισμός Cache', 2)
    
    # LiteSpeed Cache
    try:
        await send_log(log_queue, 'navigate', '🧭 Ψάχνω LiteSpeed Cache', 2)
        await page.goto(f'{wp_url}/wp-admin/admin.php?page=litespeed', wait_until='networkidle', timeout=10000)
        
        purge_button = page.locator('a:has-text("Purge All")').first
        if await purge_button.count() > 0:
            await send_log(log_queue, 'click', '🖱️ Πατάω Purge All (LiteSpeed)', 2)
            await purge_button.click()
            await asyncio.sleep(2)
            await send_log(log_queue, 'success', '✅ LiteSpeed Cache καθαρίστηκε!', 2)
        else:
            await send_log(log_queue, 'error', '⚠️ LiteSpeed Cache δεν βρέθηκε', 2)
    except:
        await send_log(log_queue, 'error', '⚠️ LiteSpeed Cache δεν είναι εγκατεστημένο', 2)
    
    # WP Super Cache
    try:
        await send_log(log_queue, 'navigate', '🧭 Ψάχνω WP Super Cache', 3)
        await page.goto(f'{wp_url}/wp-admin/options-general.php?page=wpsupercache', wait_until='networkidle', timeout=10000)
        
        delete_button = page.locator('input[value="Delete Cache"]').first
        if await delete_button.count() > 0:
            await send_log(log_queue, 'click', '🖱️ Πατάω Delete Cache', 3)
            await delete_button.click()
            await asyncio.sleep(2)
            await send_log(log_queue, 'success', '✅ WP Super Cache καθαρίστηκε!', 3)
    except:
        await send_log(log_queue, 'error', '⚠️ WP Super Cache δεν είναι εγκατεστημένο', 3)
    
    await send_log(log_queue, 'success', '🎉 Ολοκληρώθηκε ο καθαρισμός cache!', 99)

async def execute_task(task_id: str, request: TaskRequest):
    """Εκτέλεση task"""
    log_queue = asyncio.Queue()
    tasks[task_id]['log_queue'] = log_queue
    tasks[task_id]['status'] = 'running'
    
    try:
        playwright = await async_playwright().start()
        browser = await playwright.chromium.launch(
            headless=True,
            args=[
                '--disable-dev-shm-usage',
                '--disable-gpu',
                '--no-sandbox',
                '--disable-setuid-sandbox',
                '--window-size=1366,768'
            ]
        )
        
        page = await browser.new_page()
        
        # Login
        await wp_login(page, request.wp_url, request.wp_user, request.wp_pass, log_queue)
        
        # Execute specific task
        if request.task_type == 'yoast_seo':
            await task_yoast_seo(page, request.wp_url, log_queue)
        elif request.task_type == 'viva_wallet':
            await task_viva_wallet(page, request.wp_url, log_queue)
        elif request.task_type == 'clear_cache':
            await task_clear_cache(page, request.wp_url, log_queue)
        else:
            await send_log(log_queue, 'error', f'❌ Άγνωστο task type: {request.task_type}', 0)
        
        await browser.close()
        await playwright.stop()
        
        tasks[task_id]['status'] = 'completed'
        await send_log(log_queue, 'done', '🎉 Ολοκληρώθηκε!', 999)
        
    except Exception as e:
        tasks[task_id]['status'] = 'failed'
        await send_log(log_queue, 'error', f'❌ Σφάλμα: {str(e)}', 0)
        logger.error(f"Task error: {e}", exc_info=True)

async def stream_logs(task_id: str):
    """Stream logs"""
    log_queue = tasks[task_id]['log_queue']
    
    while True:
        try:
            log = await asyncio.wait_for(log_queue.get(), timeout=2.0)
            yield f"data: {json.dumps(log)}\n\n"
            
            # Αν τελείωσε, σταμάτα το stream
            if log.get('type') == 'done' or log.get('type') == 'error':
                await asyncio.sleep(1)
                break
                
        except asyncio.TimeoutError:
            # Heartbeat
            yield f"data: {json.dumps({'type': 'info', 'message': '💭 Working...', 'step': 0})}\n\n"

@app.post("/execute-stream")
async def execute_stream(request: TaskRequest, background_tasks: BackgroundTasks):
    """Start task execution"""
    task_id = str(uuid.uuid4())
    
    tasks[task_id] = {
        'status': 'pending',
        'log_queue': None
    }
    
    # Start background task
    background_tasks.add_task(execute_task, task_id, request)
    
    # Wait for log_queue to be ready
    for _ in range(10):
        await asyncio.sleep(0.5)
        if tasks[task_id]['log_queue'] is not None:
            break
    
    return StreamingResponse(
        stream_logs(task_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no"
        }
    )

@app.get("/health")
def health():
    return {"status": "ok", "service": "wordpress-automation", "version": "1.0.0-no-ai"}

@app.get("/")
def root():
    return {
        "name": "WordPress Automation API - No AI",
        "version": "1.0.0",
        "features": [
            "Yoast SEO automation",
            "Viva Wallet installation",
            "Cache clearing",
            "No AI - Pure Playwright"
        ],
        "endpoints": {
            "execute": "/execute-stream",
            "health": "/health"
        }
    }

if __name__ == "__main__":
    import uvicorn
    import os
    port = int(os.environ.get("PORT", 8000))
    logger.info(f"🚀 Starting WordPress Automation (No AI) on port {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
