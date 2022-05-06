FROM tensorflow/tensorflow:latest-gpu

#ENV VIRTUAL_ENV=/opt/venv
#RUN python -m venv $VIRTUAL_ENV
#ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --upgrade pip

WORKDIR /app
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY hebrew_diacritized hebrew_diacritized
COPY tests tests
COPY final_model final_model
COPY nakdimon nakdimon

#CMD ["python", "nakdimon", "server"]
ENTRYPOINT ["python", "nakdimon", "results"]
