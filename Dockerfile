    FROM python:3.9-slim-buster

    WORKDIR /

    # Copy the requirements file and install dependencies
    COPY requirements.txt .
    RUN pip install -r requirements.txt

    # Copy the application code into the container
    COPY . .

    # Expose the port your application listens on (if applicable)
    EXPOSE 8000
