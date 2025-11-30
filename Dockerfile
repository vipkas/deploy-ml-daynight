# BARIS 1: Base Image
FROM python:3.9-slim

# BARIS 2: Install System Dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# BARIS 3: Working Directory
WORKDIR /app

# BARIS 4: Copy Requirements
COPY requirements.txt .

# BARIS 5: Install Dependencies
RUN pip install --no-cache-dir -r requirements.txt

# BARIS 6: Copy Source Code
COPY . .

# BARIS 7: Permission
RUN chmod -R 777 /app

# BARIS 8: Expose Port
EXPOSE 7860

# BARIS 9: Command to Run
CMD ["python", "app.py"]