# Quantum Inference

### Target
Given circuit $|\psi\rangle_f = U_1 U_2 \ldots U_N|x\rangle$, with $U_k(\theta_k)=\exp{\frac{i\theta_k}{2}\Sigma_k}$,

we want to model $p(y|x) \equiv |\langle y|U_1 U_2 \ldots U_N|x\rangle|^2$.

The data set is $z\sim\pi(x,y)$.

### Gradient of Probability
$\frac{\partial p_\theta(y|x)}{\partial \theta_k} =\frac{i\theta}{2}\left[\langle x|\mathbf{U}^\dagger|y\rangle\langle y|U_{1:k} \Sigma_k U_{k+1:N}|x\rangle-\langle x|U_{k+1:N}^\dagger \Sigma_k U_{1:k}^\dagger|y\rangle\langle y|\mathbf{U}|x\rangle\right] $

define $O\equiv U_{1:k}^\dagger|y\rangle\langle y|U_{1:k}$ and $|\psi_k\rangle\equiv U_{k+1:N}|x\rangle$

we have $\frac{\partial p_\theta(y|x)}{\partial \theta_k} =\frac{i}{2}\left[\langle\psi_k|[O,\Sigma_k]|\psi_k\rangle\right] $

For arbituary operator $O$ we have $[O,\Sigma_k]= -i\left[A^\dagger OA-AOA^\dagger \right]$, with $A=\frac{1}{\sqrt{2}} (1+i\Sigma_k)$ (A is unitary by nature), this could be estimated unbiasedly.

As a special case $[O,\Sigma_k]= -i\left[U_k(\frac\pi 2)^\dagger OU_k(\frac \pi 2)-U_k(-\frac\pi 2)^\dagger OU_k(-\frac \pi 2)\right]​$, as long as $\Sigma^2=1​$.

Then,

$\frac{\partial p_\theta(y|x)}{\partial \theta_k} =\frac{1}{2}\left[p^+(y|x)-p^-(y|x)\right] $

### MMD Loss

Since we can estimate $\pi(x)$ from the dataset, training $p(x|y)\sim\pi(x|y)$ is equivalent to training $p_\theta(y|x)\pi(x)\sim \pi(x,y)$. Let's define $p(x,y)\equiv p_\theta(y|x)\pi(x)$.

Given $p(x,y)$ and $\pi(x,y)$, the MMD loss is

$\mathcal L=\langle K(z,z')\rangle_{z\sim p,z'\sim p}-2\langle K(z,z')\rangle_{z\sim p,z'\sim\pi}+{\rm const.}$

#### How to calculate loss

e.g. $\langle K(z,z')\rangle_{z\sim p,z'\sim p}$

equivalent to

$\begin{align}&\int K(z,z')p(x,y)p(x',y')dxdx'dydy'\\ =&\int K(z,z')p(x|y)p(x'|y')\pi(y)\pi(y')dxdx'dydy'\end{align}$

#### The gradient

$\begin{align}\frac{\partial L}{\partial \theta_k}=2\int K(z,z')\frac{\partial p(z)}{\partial \theta_k}p(z')dzdz'-2\int K(z,z')\frac{\partial p(z)}{\partial \theta_k}\pi(z')dzdz'\end{align}$

$$\begin{align}\frac{\partial L}{\partial \theta_k}&=\int K(z,z')\left[p^+_\theta(x|y)-p^-_\theta(x|y)\right]\pi(y)p_\theta(z')dzdz'\\&-\int K(z,z')\left[p^+_\theta(x|y)-p^-_\theta(x|y)\right]\pi(y)\pi(z')dzdz'\end{align}$$

