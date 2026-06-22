FROM python:3.10

WORKDIR /code

# Install system dependencies from packages.txt
COPY packages.txt /code/packages.txt
RUN apt-get update && xargs -a packages.txt apt-get install -y && rm -rf /var/lib/apt/lists/*

# Next, copy requirements...
COPY ./requirements.txt /code/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /code/requirements.txt

COPY . .
EXPOSE 7860
CMD ["gunicorn", "-b", "0.0.0.0:7860", "app:app"]