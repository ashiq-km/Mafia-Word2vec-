# 1. Base image: Start with a light weight Python version
FROM python:3.10-slim


# 2. Set the working directory inside the container
WORKDIR /app

# 3. Copy dependencies first (Optimization: keeps the cache if requirements don't change)
COPY requirements.txt . 

# 4. Install dependencies
# We add --no-cache-dir to keep the image small
RUN pip install --no-cache-dir -r requirements.txt


# 5. Copy the rest of the application code
COPY . .


# 6. Set environment variables
# This ensures Python doesn't buffer output (logs appear instantly)
ENV PYTHONUNBUFFERED=1



# 7. Define the default command
# When the container starts, what should it do? 
# For now, let's run the training script as a test.
CMD ["python", "-m", "src.train"]



