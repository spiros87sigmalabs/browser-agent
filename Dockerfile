FROM python:3.10-slim

ENV DEBIAN_FRONTEND=noninteractive

# Install system deps + Chromium
RUN apt-get update && apt-get install -y \
    chromium \
    chromium-driver \
    fonts-liberation \
    libnss3 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libxkbcommon0 \
    libxcomposite1 \
    libxdamage1 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    libatspi2.0-0 \
    libxshmfence1 \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Set chromium path so BrowserUse can find it
ENV BROWSER_USE_CHROMIUM_PATH=/usr/bin/chromium

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

EXPOSE 8080

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080"]
