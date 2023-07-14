## SVM-Poisoning

Support Vector Machines (SVMs) are a popular type of machine learning algorithm used in many applications, including image classification, natural language processing, and anomaly detection. However, like many machine learning algorithms, SVMs are vulnerable to attacks that manipulate their training data.
One such attack is called a poisoning attack, where an attacker adds malicious data to the training set with the goal of manipulating the SVM's decision boundary. Poisoning attacks can be targeted, where the attacker wants the SVM to classify specific inputs incorrectly, or untargeted, where the attacker just wants to reduce the SVM's overall accuracy.


In this lab, we will explore the effects of a poisoning attack on an SVM trained on a dataset of handwritten digits. We will use a simple gradient ascent algorithm to craft poison samples that can fool the SVM into misclassifying digits. We will evaluate the effectiveness of the attack by measuring the SVM's accuracy on a test set with and without the poison samples.
By the end of this lab, you will have a better understanding of how SVM poisoning attacks work and how to defend against them. You will also gain experience working with SVMs and gradient ascent algorithms in Python.
## Target Audience

### Instructors

If you are an instructor teaching cybersecurity concepts, you can use this example to provide a better understanding of
poisoning attacks as well as understand binary classifiers and mathematical models used in machine learning. You can also modify the
features used for the machine learning training and gauge the accuracy of the model.

### Students

If you are a student in a cybersecurity class, or, a budding programmer with an interest in the fields of cybersecurity or machine learning,
this notebook provides an easy way to explore poisoning attacks against simple ML models like SVM, which can be further explored for Neural netwrok models.

## Design and Architecture

This demonstration is designed as a single Docker container that hosts a jupyter notebook. The jupyter notebook contains instructive material
on how to read, clean and explore the data. Instructions on selecting features, building the SVM model and printing evaluation
reports are also contained in the jupyter notebook.

## Installation and Usage

The recommended approach to running this container is on CHEESEHub, a web platform for cybersecurity demonstrations. CHEESEHub
provides the necessary resources for orchestrating and running containers on demand. In order to set up this application to be
run on CHEESEHub, an *application specification* needs to be created that configures the Docker image to be used, memory and
CPU requirements, and, the ports to be exposed. The JSON *spec* for this SQL Injection demonstration can be found here **(FIXME)**

CHEESEHub uses Kubernetes to orchestrate its application containers. You can also run this application on your own Kubernetes
installation. For instructions on setting up a minimal Kubernetes cluster on your local machine, refer to minikube **(FIXME)**

Before being able to run on either CHEESEHub or Kubernetes, a Docker image needs to be built for this application container.
The container definition can be found in the *Dockerfile* in this repository. To build the container, run:

``
docker build -t <image tag of your choice> .
``

Once a Docker image has been built, you can run this container using just the Docker engine:

``
docker run -d -p 8888 <image tag from above>
``

Since the user facing interface of the container is the Jupyter notebook we expose port 8888 to be accessible on the host machine.

### Usage
On navigating to the URL of the container in your browser, you will be presented with a Jupyter notebook interface displaying a list of files
and folders. Click to select the svm-poisoning.ipyb Jupyter notebook to begin. The notebook includes a step-by-step overview of how to conduct
the poisoning attacks against ML models. The code can be run in the code cells within the jupyter                            
notebook using the "run" command.

## How to Contribute

To report issues or contribute enhancements to this application, open a GitHub issue.
