FROM tensorflow/tensorflow:latest-gpu

#ENV VIRTUAL_ENV=/opt/venv
#RUN python -m venv $VIRTUAL_ENV
#ENV PATH="$VIRTUAL_ENV/bin:$PATH"
RUN pip install --upgrade pip

WORKDIR /app
COPY README.md README.md
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY hebrew_diacritized hebrew_diacritized
COPY tests tests
COPY models models
COPY nakdimon nakdimon

RUN chown -R 1000:1000 .
RUN chmod -R 755 .

CMD python nakdimon run_test \
 && python nakdimon results --systems MajAllWithDicta Snopi Morfix Dicta Nakdimon
