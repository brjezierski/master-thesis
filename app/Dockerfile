FROM continuumio/miniconda3

WORKDIR /app

SHELL ["/bin/bash", "--login", "-c"]

# Create the environment:
COPY environment.yml .
RUN conda env create -f environment.yml
RUN pip uninstall --yes werkzeug
RUN pip install -v https://github.com/pallets/werkzeug/archive/refs/tags/2.0.3.tar.gz

RUN conda init bash

# Activate the environment, and make sure it's activated:
RUN echo "conda activate tm" > ~/.bashrc
RUN echo "Make sure flask is installed:"
RUN python -c "import flask"

# The code to run when container is started:
COPY . .
ENTRYPOINT python app.py