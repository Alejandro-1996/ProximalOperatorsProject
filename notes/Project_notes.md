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
where $y$ corresponds to the measurements, $h$ corresponds to an operator (e.g. the Point Spread Function in a $2D$ imaging problem) and *w* is simply noise. In this case, we have complete access to $y$, we should know the operator $h$ and, $w$ the white noise, can be approximated, usually with a Gaussian.  

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
