# state_of_art_cnns
Repository with recent state-of-the-art CNN architectures implemented in Keras.

Each architecture can be tested out-of-the-box with the tester.py script using:  
  python tester.py <model_name> [--dropout [<dropout_rate>]] 
		[--droppath [<droppath_rate>]] [--lr [<learning_rate>]] 
		[--epochs [<num_of_training_epochs>]]

The test script reqiures the "dl_utilities" package (easily attainable using the "set_up.sh" shell script).
		
Like in the test script, it is possible to initiate new models and train them by simply importing the needed code.

In other words, each directory should be able to act as its own Python package by importing using:
  from directory_name import filename as <desired_model_name_handle>

This is why each directory has its own empty "__init__.py" file.

Though these packages should be usable with any Keras backend, they have only been tested with a TensorFlow backend.
