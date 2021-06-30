# Proximal Operators for Nonnegative Inverse Problems

Repository of the course *Microengineering project I* (10 credits), a [semester research project](http://bigwww.epfl.ch/teaching/projects/current.html#id_4700) carried out from February to June 2020 at EPFL's [Biomedical Imaging Group](http://bigwww.epfl.ch/). 

The project was supervised by:
* [Pol del Aguila Pla](https://poldap.github.io), (pol.delaguilapla@epfl.ch, [poldap](https://github.com/poldap))

### Scope of the Project

The project is dedicated to the study of image regularizers and their combination with nonnegativity constraints. This is in the context of solving imaging problems, whose algorithms of choice take advantage of the proximal operators of nonnegativity constraints and of the regularizers. However, this involves costly variable splitting, and additional optimzation problems. This splitting could be avoided by fulfilling any of the two conditions

<img src="https://user-images.githubusercontent.com/65513243/123969976-04583980-d9b9-11eb-87b5-7f2d040f41ce.png" alt="alt text" width="300">

where <img src="https://render.githubusercontent.com/render/math?math=\mathcal{R}"> denotes the image regularizer and <img src="https://render.githubusercontent.com/render/math?math=\delta"> denotes the nonnegativity constraints. The left hand side of the equations represent the ground truth.

Therefore, the goal of this project is to project is to experimentally test the previous equations for different common image regularizers, and build a data base of which regularizers are candidates for reduced splitting, and therefore can potentially improve performance. 

### Methodology

Given that all image regularizers are convex, the methodology was to use [CVXPy](https://www.cvxpy.org/index.html), a Python-embedded modeling language for convex optimization problems, to obtain both the ground truth and the right-hand-side of the tested equations. Even though this method incurrs in inherent numerical errors, it has the advantage that many different regularizers can be studied through a common framework, without the need to use proximal analysis to obtain the ground truth and the proximal operators of regularizers.  

### Code and Directory Structure

The respository has 5 subdirectories
 * [`Code`](https://github.com/Alejandro-1996/ProximalOperatorsProject/tree/master/code): Contains the file [`proximal.py`](https://github.com/Alejandro-1996/ProximalOperatorsProject/tree/master/code/proximal.py), with the framework to test for reduced splitting. Moreover, it contains the Jupyter notebooks [`Evaluations.ipynb`](https://github.com/Alejandro-1996/ProximalOperatorsProject/tree/master/code/Evaluations.ipynb) -where all the different image regularizers are tested- [`Experiments.ipynb`](https://github.com/Alejandro-1996/ProximalOperatorsProject/tree/master/code/Experiments.ipynb) -where some of the results are tested on real images, on an image denoising task- and [`Plots.ipynb`](https://github.com/Alejandro-1996/ProximalOperatorsProject/tree/master/code/Plots.ipynb) -where some relevant plots for the report are created-. The rest are utils for the code to run. See the wiki for details.
 * [`Data`](https://github.com/Alejandro-1996/ProximalOperatorsProject/tree/master/data): Contains the file `peppers.tiff`, the only image necessary for the experiments (the rest is imported from the [`skimage.data`](https://scikit-image.org/docs/dev/api/skimage.data.html) module). Furthermore, it contains the results from the experiments in `.npy` formal. The [`README.txt`](https://github.com/Alejandro-1996/ProximalOperatorsProject/tree/master/README.txt) explains this files in more detail
 * [`Presentation`](https://github.com/Alejandro-1996/ProximalOperatorsProject/tree/master/presentation): Contains the file [`Final_PPT_Nogueron.pdf`](https://github.com/Alejandro-1996/ProximalOperatorsProject/blob/master/presentation/Final_PPT_Nogueron.pdf), the final graded  presentation of the project in PDF format.
 *  [`Report`](https://github.com/Alejandro-1996/ProximalOperatorsProject/tree/master/report): Contains the file [`Report_ANogueron.pdf`](https://github.com/Alejandro-1996/ProximalOperatorsProject/blob/master/report/Report_ANogueron.pdf), the final report in PDF format. Moreover, it contains the directory [`LaTex_Project`](https://github.com/Alejandro-1996/ProximalOperatorsProject/tree/master/report/LaTex_Project), with all the files necessary to generate the report in *LaTex*.
 *  [`Web`](https://github.com/Alejandro-1996/ProximalOperatorsProject/tree/master/web): Contians the file [`abstract`](https://github.com/Alejandro-1996/ProximalOperatorsProject/tree/master/web/abstract.txt) and the image [`TV_Experiment`](https://github.com/Alejandro-1996/ProximalOperatorsProject/tree/master/web/TV_Experiment.jpg), that briefly illustrate and explain the project.

### Results 
The main results are:
| Regularizer  | Reduced Splitting |
| ------------- | ------------- |
| *p* norms  | Yes  |
| *1*-dimensional TV   | Yes  |
| nonisometric TV (*2D*)  | Yes  |
| isometric TV (*2D*)  | No  |
| Group Sparsity  | Yes  |
| Hessian-Schatten norm  | ?  |

For detailed results and discussion, look at section 4 of the [report](https://github.com/Alejandro-1996/ProximalOperatorsProject/blob/master/report/Report_ANogueron.pdf).
