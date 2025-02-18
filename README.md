# PyDFTsalr
An python library for calculations using the classical Density Functional Theory (cDFT) for SALR (Short-Range Attraction and Long-Range Repulsion) fluids in 3D geometries.

## Dependencies

* [NumPy](https://numpy.org) is the fundamental package for scientific computing with Python.
* [PyTorch](https://pytorch.org/) is a high-level library for machine learning, with multidimensional tensors that can also be operated on a CUDA-capable NVIDIA GPU. 
<!-- * [Matplotlib](https://matplotlib.org/stable/index.html) is a comprehensive library for creating static, animated, and interactive visualizations in Python.
* *Optional*: [SciencePlots](https://github.com/garrettj403/SciencePlots) is a Matplotlib styles for scientific figures -->

## Installation

### Option 1: Using `setup.py`

Clone `PyDFTsalr` repository if you haven't done it yet.

```Shell
git clone https://github.com/elvissoares/PyDFTsalr
```

Go to `PyDFTsalr`'s root folder, there you will find `setup.py` file, and run the command below:

```Shell
pip install -e .
```

The command `-e` permits to edit the local source code and add these changes to the pydftlj library.

### Option 2: Using pip to install directly from the GitHub repo

You can run

```Shell
pip install git+https://github.com/elvissoares/PyDFTsalr
```

and then you will be able to access the pydftlj library.

## cDFT basics

The cDFT is the extension of the equation of state to treat inhomogeneous fluids. For a fluid with temperature T, total volume V, and chemical potential $\mu$ specified, the grand potential, $\Omega$, is written as

$$\Omega[\rho(\boldsymbol{r})] = F[\rho (\boldsymbol{r})] +  \int_{V} [ V^{(\text{ext})}(\boldsymbol{r}) - \mu ]\rho(\boldsymbol{r}) d\boldsymbol{r}$$

where $F[\rho (\boldsymbol{r})] $ is the free-energy functional, $V^{(\text{ext})} $ is the external potential, and $\mu $ is the chemical potential. The free-energy functional  can be written as a sum $ F = F^\text{id} + F^\text{exc} $, where $F^\text{id} $ is the ideal gas contribution and $F^\text{exc}$ is the excess contribution.

The ideal-gas contribution $F^\text{id} $ is given by the exact expression

$$ F^{\text{id}}[\rho (\boldsymbol{r})] = k_B T\int_{V} \rho(\boldsymbol{r})[\ln(\rho (\boldsymbol{r})\Lambda^3)-1] d\boldsymbol{r}$$

where $k_B $ is the Boltzmann constant, and $\Lambda $ is the well-known thermal de Broglie wavelength.

The excess Helmholtz free-energy, $F^{\text{exc} }$, is the free-energy functional due to particle-particle interactions and can be splitted in the form

$$ F^{\text{exc}}[\rho (\boldsymbol{r})] = F^{\text{hs}}[\rho (\boldsymbol{r})] + F^{\text{att}}[\rho (\boldsymbol{r})] $$
where $F^{\text{hs}} $ is the hard-sphere repulsive interaction excess contribution and $F^{\text{tail}} $ is the tail interaction excess contribution. 

The hard-sphere contribution, $F^{\text{hs}} $, represents the hard-sphere exclusion volume correlation and it can be described using different formulations of the fundamental measure theory (FMT) as **W**hite **B**ear version **II** (**WBII**) - [Hansen-Goos, H. & Roth, R. J., Phys. Condens. Matter 18, 8413–8425 (2006)](https://iopscience.iop.org/article/10.1088/0953-8984/18/37/002)

The tail contribution, $F^\text{tail}$, can be described by the Random Phase Approximation as 
$$ F^{\text{tail}}[\rho (\boldsymbol{r})] = \iint \rho (\boldsymbol{r})u (|\boldsymbol{r}-\boldsymbol{r}'|)\rho (\boldsymbol{r}') \text{d}\boldsymbol{r}\text{d}\boldsymbol{r}'$$
where the pair potential $u(r)$ can be modeled as a double-Yukawa potential, a double-gaussian potential, a double-step potential. 

The *equilibrium* is given by the functional derivative of the grand potential in the form 

$$ \frac{\delta \Omega}{\delta \rho(\boldsymbol{r})} = k_B T \ln(\rho(\boldsymbol{r}) \Lambda^3) + \frac{\delta F^{\text{exc}}[\rho]}{\delta \rho(\boldsymbol{r})}  +V^{(\text{ext})}(\boldsymbol{r})-\mu = 0$$

The *dynamics* is given by the DDFT equation in the form 
$$\frac{\partial \rho}{\partial t} = \nabla \cdot \left[ D \rho(\boldsymbol{r}) \nabla \left(\frac{\delta \beta F[\rho]}{\delta \rho}\right) \right]$$


<!-- # Cite PyDFTsalr

If you use PyDFTsalr in your work, please consider to cite it using the following reference:

Soares, Elvis do A, Amaro G Barreto, and Frederico W Tavares. 2023. “Classical Density Functional Theory Reveals Structural Information of H2 and CH4 Fluids Adsorbed in MOF-5.” [Fluid Phase Equilibria](https://doi.org/10.1016/j.fluid.2023.113887), July, 113887.   ArXiv: [2303.11384](https://arxiv.org/abs/2303.11384)

Bibtex:

    @article{Soares2023, 
    author = {Soares, Elvis do A and Barreto, Amaro G and Tavares, Frederico W}, 
    doi = {10.1016/j.fluid.2023.113887}, 
    issn = {03783812}, 
    journal = {Fluid Phase Equilibria}, 
    keywords = {Adsorption,Density functional theory,Metal–organic framework,Structure factor}, 
    month = {jul}, 
    pages = {113887}, 
    title = {{Classical density functional theory reveals structural information of H2 and CH4 fluids adsorbed in MOF-5}}, 
    url = {https://linkinghub.elsevier.com/retrieve/pii/S037838122300167X}, 
    year = {2023} 
    }  -->


# Contact
Elvis Soares: elvis@peq.coppe.ufrj.br

Programa de Engenharia Química - PEQ/COPPE

Universidade Federal do Rio de Janeiro
