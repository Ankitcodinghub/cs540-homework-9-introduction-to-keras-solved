# cs540-homework-9-introduction-to-keras-solved
**TO GET THIS SOLUTION VISIT:** [CS540 Homework 9-Introduction to Keras Solved](https://www.ankitcodinghub.com/product/cs540-hw9-introduction-to-keras-solved/)


---

📩 **If you need this solution or have special requests:** **Email:** ankitcoding@gmail.com  
📱 **WhatsApp:** +1 419 877 7882  
📄 **Get a quote instantly using this form:** [Ask Homework Questions](https://www.ankitcodinghub.com/services/ask-homework-questions/)

*We deliver fast, professional, and affordable academic help.*

---

<h2>Description</h2>



<div class="kk-star-ratings kksr-auto kksr-align-center kksr-valign-top" data-payload="{&quot;align&quot;:&quot;center&quot;,&quot;id&quot;:&quot;117944&quot;,&quot;slug&quot;:&quot;default&quot;,&quot;valign&quot;:&quot;top&quot;,&quot;ignore&quot;:&quot;&quot;,&quot;reference&quot;:&quot;auto&quot;,&quot;class&quot;:&quot;&quot;,&quot;count&quot;:&quot;1&quot;,&quot;legendonly&quot;:&quot;&quot;,&quot;readonly&quot;:&quot;&quot;,&quot;score&quot;:&quot;5&quot;,&quot;starsonly&quot;:&quot;&quot;,&quot;best&quot;:&quot;5&quot;,&quot;gap&quot;:&quot;4&quot;,&quot;greet&quot;:&quot;Rate this product&quot;,&quot;legend&quot;:&quot;5\/5 - (1 vote)&quot;,&quot;size&quot;:&quot;24&quot;,&quot;title&quot;:&quot;CS540  Homework 9-Introduction to Keras Solved&quot;,&quot;width&quot;:&quot;138&quot;,&quot;_legend&quot;:&quot;{score}\/{best} - ({count} {votes})&quot;,&quot;font_factor&quot;:&quot;1.25&quot;}">

<div class="kksr-stars">

<div class="kksr-stars-inactive">
            <div class="kksr-star" data-star="1" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="2" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="3" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="4" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" data-star="5" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>

<div class="kksr-stars-active" style="width: 138px;">
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
            <div class="kksr-star" style="padding-right: 4px">


<div class="kksr-icon" style="width: 24px; height: 24px;"></div>
        </div>
    </div>
</div>


<div class="kksr-legend" style="font-size: 19.2px;">
            5/5 - (1 vote)    </div>
    </div>
Changelog

11/17: Evaluate model output details highlighted

11/18: More explanation about the print_stats()

11/18: Evaluate model’s output loss should be formatted with four decimal places (as shown in the sample output).

11/18: Not test the model.metrics_names output

11/20: Change the example in train_model function from “s/example” to “s/step” and removed “Train on X samples” to conform with the CSL machine output

Assignment Goals

Get TensorFlow (and Keras) set up for your environment

Familiarize yourself with the tools

Perform some basic neural network tasks using TF’s utilities Happy deep learning! 🙂

Summary

just SciPy and NumPy to support your projects, as their utilities have been developed by a team of professionals and undergo rigorous testing and verification.

In this homework, we’ll be exploring the Keras (https://www.tensorflow.org/guide/keras) package in TensorFlow and its uses in deep learning.

Part 1: Setting up the Python Virtual Environment

In this assignment, you will familiarize yourself with the Python Virtual Environment. Working in a virtual environment is an important part of working with modern ML platforms, so we want you to get a flavor of that through this assignment. Why do we prefer virtual environments? Virtual environments allow us to install packages within the virtual environment without affecting the host system setup. So you can maintain project-specific packages in respective virtual environments.

We suggest that you use the CS lab computers for this homework. You can also work on your personal system for the initial development, but finally, you will have to test your model on the CSL lab computers.

Find more instructions: How to access CSL Machines Remotely (https://csl.cs.wisc.edu/)

(https://csl.cs.wisc.edu/) The following are the installation steps for Ubuntu/MacOS (CSL machines are recommended). For Windows installation steps refer to this.

(https://www.tensorflow.org/install/pip#windows_1)

1. You will be working on Python 3 as Tensorflow is not supported on Python 2.

Read more about Tensorflow and Python version here (https://www.tensorflow.org/install) .

2. To check your Python version use:

If you have an alias set for python=python3 then both should show the same versions (3.x.x) 3. Setup a Python Virtual Environment (https://www.tensorflow.org/install/pip#ubuntu-macos) :

Here the name of our virtual environment is Tensorflow (you can use any other name if you want).

4. Activate the environment:

5. From your virtual environment shell, run the following commands to successfully upgrade pip andinstall Tensorflow:

i i t ll d t fl

Please make sure that you have the same version of TensorFlow as installed as us. You can check the version of the packages installed using the following command:

6. For deactivating the virtual environment:

7. In your implementation you will be using Tensorflow and Keras by importing them as follows:

Deliverable for Part 1:

Once you have set-up the virtual environment, please run the following command:

This command will dump the output of “pip freeze” into a .txt file called setup_output.txt. We will be checking that you have installed the correct version of TensorFlow.

Part 2: Program Specification

In this program, you will be using Tensorflow Keras to build a simple deep learning model for predicting labels of images of handwritten images. You will learn how to build, train, evaluate models, and make predictions on test data using this model This program asks you to implement the following functions in predictions on test data using this model. This program asks you to implement the following functions in Python.

1. get_dataset(training=True) —

Input: an optional boolean argument (default value is True for training dataset)

Return: two NumPy arrays for the train_images and train_labels

2. print_stats(train_images, train_labels) — This function will print several statistics about the data Input: the dataset and labels produced by the previous function; does not return anything

3. build_model() — takes no arguments and returns an untrained neural network model

4. train_model(model, train_images, train_labels, T) — takes the model produced by the previous function and the dataset and labels produced by the first function and trains the data for T epochs; does not return anything

5. evaluate_model(model, test_images, test_labels, show_loss=True) — takes the trained model produced by the previous function and the test image/labels, and prints the evaluation statistics as described below (displaying the loss metric value if and only if the optional parameter has not been set to False); does not return anything

6. predict_label(model, test_images, index) — takes the trained model and test images, and prints the top 3 most likely labels for the image at the given index, along with their probabilities; does not return anything

You are free to implement any other utility function. But we will only be testing the functionality using the above 6 APIs, so make sure that each of them follows the exact function signature and returns. You can also use helper methods to visualize the images from the MNIST dataset for a better understanding of the dataset and the labels. But it is totally optional and does not carry any points.

Get Dataset

Unlike last time, we’re not going to make you curate your own dataset — or even download it. Keras contains a few datasets from the National Institute of Standards and Technology (NIST). Typically the

“hello world” of datasets is a bunch of images of handwritten numbers, MNIST (http://yann.lecun.com/exdb/mnist/) which we will be using for this homework.

For this function, you’ll want to use the load_data() function on keras.datasets.mnist to retrieve and return the 2D array of integers and labels representing images of handwritten numbers:

Train images and labels are the data we’ll be using to train our neural network; test images and labels are what we’ll be using to test that model.

When the optional argument is unspecified, return the training images and labels as NumPy arrays. If the optional argument is set to False return the testing images and labels This function should be the optional argument is set to False, return the testing images and labels. This function should be called like so:

Dataset Statistics

This function explores the datasets and labels produced by the previous function.

Note that the labels are themselves just integers, so will want to make use of the following label translations here:

Print out the following statistics for the data provided in the arguments:

1. Total number of images in the given dataset

2. Image dimension

3. Number of images corresponding to each of the class labels

For example, the call for print_stats on the training data should output something like this. Similarly, you will be able to get the statistics of the test dataset as well. (note: these numbers are made up):

Your code should work for both the training and testing data, and produce different results for each.

Build Model

For this assignment, we’ll be using some of the core Keras layers (https://keras.io/layers/core/) (that’s a good resource to check on for layer specifics): Flatten and Dense.

To hold these layers, begin by creating a Sequential (https://keras.io/getting-started/sequential-modelguide/) object (that’s ALSO a good resource for using Keras models):

and then add the following layers in this order:

1. A Flatten layer to convert the 2D pixel array to a 1D array of numbers; you’ll need to specify the input shape here based on your results from print_stats() above.

2. A Dense layer with 128 nodes and a ReLU activation. (https://keras.io/activations/)

3. A Dense layer with 64 nodes and a ReLU activation.

4. A Dense layer with 10 nodes.

The Flatten layer just serves to reformat the data, but you’ll have to train the Dense layers’ parameters with the training data before we can use the model.

The model should be compiled using the following params:

Optimizer (https://keras.io/optimizers/) : SGD with a learning rate of 0.001.

Loss Function (https://keras.io/losses/) : Sparse categorical cross-entropy (with the from_logits parameter set to True).

Metric (https://keras.io/metrics/) : Accuracy.

Finally, return the compiled model.

&gt;&gt;&gt; print(model.optimizer)

(If you don’t compile your model, you’ll still see the Sequential line, but the other three will not work.)

Train Model

This function is mostly a wrapper function. Use the Keras-provided model.fit() function with the training images and labels, with the number of epochs from the parameters, and let Keras do the rest!

This sample output is performed on a subset of the data (for simplicity); your output will be different:

Evaluate Model

Now that you’ve trained the model, you will want to evaluate how good it is. Keras also provides a function to run a test dataset and compare the results to its true labels; again you’ll be writing what is effectively a wrapper function for the model.evaluate(). Please use the function’s optional verbose parameter to suppress the default output.

The evaluate() function provides two outputs:

Format the accuracy output with two decimal places and the accuracy should be shown as a percentage. Format the loss with four decimal places. (As shown in the sample output below)

Predict Label

Run the model.predict() function using the test images, and display the top three most likely class labels for the image at the given index (assumed to be valid) along with their respective probabilities (again, your output will vary in its exact numbers but should be similar):

Deliverable for Part 2:

A single file named intro_keras.py containing the methods mentioned in the program specification section.

Submission Notes

Please submit your files in a zip file named hw9_&lt;netid&gt;.zip, where you replace &lt;netid&gt; with your netID

(your wisc.edu login). Inside your zip file, there should be two files named: intro_keras.py and setup_output.txt. Do not submit a Jupyter notebook .ipynb file. All code should be contained in functions OR under a

check so that it will not run if your code is imported to another program.

Be sure to remove all debugging output before submission. Failure to remove debugging output will

gg g p gg g p

be penalized (10pts).

Introduction to Keras

Criteria Ratings Pts

get_dataset() returns the correct values

This encompasses both type and contents of the return value and usage of an optional argument. 20.0 pts

Full

No

print_stats() prints correct values in correct format 10.0 pts

Full

No

build_model() returns an untrained, compiled Sequential object with the specified components 10.0 pts

Full

No

train_model() trains the model with the provided dataset for the specified epochs

Note: this should produce different results based on whether the provided dataset is the full training data or a random sample 10.0 pts

Full

No

evaluate_model() displays correctly formatted output 10.0 pts Full

No

evaluate_model() displays different output based on its optional argument 5.0 pts

Full

No

predict_label() displays correctly formatted output 10.0 pts

Full

No

predict_label() displays expected top-3 categories 10.0 pts

Full

No

predict_label() displays probabilities, not logits 5.0 pts

Full

No

Criteria Ratings Pts

Python Virtual Environment setup 10.0 pts

Full

No
