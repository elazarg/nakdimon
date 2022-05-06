FROM tensorflow/tensorflow:latest-gpu
WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt
COPY nakdimon nakdimon
COPY final_model models
COPY tests tests
WORKDIR /app/nakdimon
CMD [ "python3", "nakdimon", "server"]
ENTRYPOINT [ "python3" ]
