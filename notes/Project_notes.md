# Semester Project Notes
*autor*
## Alejandro Noguerón
*alejandro.nogueronaramburu@epfl.ch*
*supervised by*:
## Dr. Pol Del Aguila Pla,
## Prof. Michael Ünser

## <a name="Index"></a> Index
1. [Introduction to the Project](#1.Intro)
2. [Code Notes](#2.Code)
3. [To Do](#3.Todo)
4. [Doubts](#3.TDoubts)

## <a name="1.Intro"></a> 1. Introduction to the Project

This section is meant to be a compendium of theoretical bases of the proect, paired with a description of it. 

An imaging problem takes the following form:
$$y[m, n] = (h[m', n']\circledast x[m', n'])[m, n] + w[m, n]$$
or in a shorter notation
$$y = Hx + w$$ 
where $y$ corresponds to the measurements, $h$ corresponds to an operator (e.g. the Point Spread Function in a $2D$ imaging problem) and *w* is simply noise. In this case, we have complete access to $y$, we should know the operator $h$ and, $w$ the white noise, can be approximated, usually with a Gaussian.  

Poor man's solution:
$$\min_{x\in \rm I\!R^N}\|Hx - y\|^2_2$$

Next generetion solution:
$$\min_{x\in \rm I\!R^N} \|Hx - y\|^2_2 + \mathcal{R}(Lx) + \delta_{\rm I\!R_+^N}$$

#### Proximal Operator
$$prox_f(v) = argmin_x(f(x)+\frac{1}{2}||x - v||_2^2)$$

## <a name="3.Todo"></a> 3. To do 
[Back To Index](#Index)


## <a name="4.Doubts"></a> 4. Doubts 
[Back To Index](#Index)

* Factor $\lambda$? Does it play a role?
* How do we manage the approach cvxpy vs analytical functions?
* $$\mathrm{prox}_f(v) = \arg \min_{x \in \rm I\!R_+^N}(\|Qx\|_1+\frac{1}{2}||x - v||_2^2)$$



$\|Lx\|_2 \mathcal{R}$

Dónde $LL^T = \alpha I$ 

L puede ser finite differences, FT, DCT, filtros como LoG (segunda derivada)(ver pag 130 prox_alg, sección 2.2, referencias). Ver sección 2.2 Boyd Proximal Algorithms. Intentar para versiones *fáciles*


## <a name="4.Results"></a> 5. Results 

| $f(x)$ |  $\mathrm{prox}_{\delta}(\mathrm{prox}_{\mathcal{R}})$  | $\mathrm{prox}_{\mathcal{R}}(\mathrm{prox}_{\delta})$ |
|----------|-------------|------|
| $\|x\|_0$ |  Yes | Yes |
| $\|x\|_1$ |  Yes | Yes |
| $\|x\|_2$ |    No   |   Yes |
| $\|DCTx\|_1$ | No |    No |
| $\|DCTx\|_2$ | No |    No |
| $\|Qx\|_1$ ($Q$: Finite Differences) | Yes |  No |
| $\|Qx\|_1$ ($Q$: LoG) | No |  No |

## <a name="4.Results"></a> 5. Lit Review

Lit review preliminary notes:

* [Unser, 2012, Wavelets, Sparsity and Biomedical Image Reconstruction](http://bigwww.epfl.ch/tutorials/unser_wavelets_sparsity2012.pdf)
    - Plots closed form solution of $\mathrm{prox}_{|\cdot|^p}$ (is this actually a norm?) without any further explanation (slide 28). 
    - Cites [Combette-Pesquet, SIAM, 2007](https://epubs.siam.org/doi/pdf/10.1137/060669498)(See below)
    
* [Proximity Operator Repository](http://proximity-operator.net/scalarfunctions.html)
    - Cites [Combettes & Nguyen, 2016](https://arxiv.org/abs/1505.00362)
    
* [Combettes & Nguyen, 2016](https://arxiv.org/abs/1505.00362)
    - ... Haven't taken the time to read 
    - Proposes a method that I cannot reproduce

* [Combette-Pesquet, SIAM, 2007](https://epubs.siam.org/doi/pdf/10.1137/060669498)
     - Gives closed form solutions for p in [1, 4/3, 3/2, 2, 3, 4] (Sketchy result for p == 2), in example 2.7
     
     
