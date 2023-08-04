FROM jupyter/base-notebook:python-3.9

# Create a new directory for Jupyter Notebook
RUN pip install -U setuptools


COPY  ./ $HOME/
WORKDIR $HOME/

RUN pip install -r requirements.txt


RUN pip install jupyter_contrib_nbextensions

RUN jupyter contrib nbextension install --user
# Expose port 8888 for Jupyter Notebook
EXPOSE 8888

# Run Jupyter Notebook on container start
CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token="]
