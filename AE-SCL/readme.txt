AE-SCL-SR
This algorithm is used to find a shared low dimentional representation in order to overcome the domain adaptation problem
you can read more ealborated explanation in the article "Neural Structural Correspondence Learning for Domain Adaptation".
 




INSTALLATION

AE-SCL-SR requires the following packages:
Python >= 2.7
numpy
scipy
Theano 
keras
scikit-learn

make sure that the DATA directory is in the same directory as the "run.py"
make sure that all the scripts are in the same directory
you can find the results, word embeddings, model's weights and more in the directory source-target that will be created 
after you run the experiments.

If you wish to run the program on a gpu device you should use the  THEANO_FLAGS=device=gpu,floatX=float32 python run.py command

DATA
Don't forget to unzip dvdUN.rar in data\dvd.
the scripts assume a XML file (like blitzer provides in his website, with less attributes) in the following structure:
<reviews>
<review>
</review>
	.
	.
	.
<review>
</review>
</reviews>

you need to store your data in "data" directory in which each directory will be named after it's domain, for example, in the "run.py" file 
we adapt from "books" to "kitchen", so we have 2 directories in the script's directory, "data\books" and "data\kitchen"
each one contains "negative.parsed" file in which we store the reviews with negative labels and "positive.parsed" for the positive ones.
we assume 1000 reviews for each file, total of 2000 labeled reviews for each domain.
In addition in each directory we need an unlabeled examples file, for kitchen for examples the file will be called "kitchenUN.txt" (the domain's name followed by UN.txt in the domain directory)

