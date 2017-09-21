# state_of_art_cnns
Repository with recent state-of-the-art CNN architectures implemented in Keras.

Each architecture has a main routine for immediately running it out of the box (with TF as a backend) using:  
  python <model_file>.py

Alternatively, it is possible to initiate new models and train them by simply importing the code to another Python file.

In other words, each directory should be able to act as its own Python package by importing using:
  import <directory_name>.<filename> as <desired_model_name_handle>

This is why each directory has its own empty "__init__.py" file.

Also, though theoretically these packages should be usable with any Keras backend, they have only been tested with Tensorflow.
