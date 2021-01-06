Instructions to run on Colab:

	From a blank cell in colab, go to the Tools menu and select "Command palette".
	This will pull up a console.
	Enter the command "Mount Drive"
	Then, in a code window, run the following code:
		from google.colab import drive
		drive.mount('/content/drive')
	Complete the authentication process to grant Colab access to your drive.
	Now, drag and drop the contents of this folder into the Colab Notebooks folder in your Google Drive.
	All of the A3 code will now be accessible to you via Colab, and you can run the test programs to verify the results.
	Enter a code snippet of the format "!python3 "/content/drive/My Drive/Colab Notebooks/code/test.py", substituting the name of any of the test files for "test.py".
	The following is a list of the files, sorted by their purpose.

List of Files:	

	Classifier
		cnn.py
		
	Test Code
		predict_test.py