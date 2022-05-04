FROM tensorflow/tensorflow:latest-gpu
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY nakdimon nakdimon
COPY final_model final_model
COPY tests testsls
WORKDIR /app/nakdimon
CMD [ "python3", "-m" , "flask", "run"]
