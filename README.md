[![Build Status](https://travis-ci.org/jmboehm/Alpaca.jl.svg?branch=master)](https://travis-ci.org/jmboehm/Alpaca.jl) [![Coverage Status](https://coveralls.io/repos/jmboehm/Alpaca.jl/badge.svg?branch=master&service=github)](https://coveralls.io/github/jmboehm/Alpaca.jl?branch=master) [![codecov.io](http://codecov.io/github/jmboehm/Alpaca.jl/coverage.svg?branch=master)](http://codecov.io/github/jmboehm/Alpaca.jl?branch=master)

# Alpaca.jl

Julia wrapper for the [alpaca R library](https://github.com/amrei-stammann/alpaca) to estimate generalized linear model with high-dimensional fixed effects.

Alpaca.jl is currently targeting alpaca 0.3.1.

## Installation

You need to have R, and Julia of course, preinstalled, but if either is 64-bit the oher needs to match (so e.g. 32-bit julia snap installation may not work, and you'll get ERROR: ... wrong ELF class: ELFCLASS64):

```
] add Alpaca

and in R:

> install.packages('alpaca')
```

## Usage

Use the `feglm` function. See the following example:

```julia
using Alpaca, RDatasets, Distributions, Random

# setting up the example data
rng = MersenneTwister(1234)
df = dataset("datasets", "iris")
df.binary = vec(Float64.(rand(rng,0:1,size(df,1),1)))
df[!,:SpeciesDummy] = categorical(df[!,:Species])
idx = rand(rng,1:3,size(df,1),1)
a = ["A","B","C"]
df.Random = vec([a[i] for i in idx])
df.RandomCategorical = categorical(df[!,:Random])

# estimating the model
result = feglm(df, @formula(binary ~ SepalWidth), Binomial(),
    fe = :(SpeciesDummy + RandomCategorical),
    vcov = :(cluster(SpeciesDummy + RandomCategorical))
    )
```

The full form of the `feglm` function is
```julia
function feglm(df::AbstractDataFrame, f::FormulaTerm,
    family::UnivariateDistribution;
    fe::Union{Symbol, Expr, Nothing} = nothing,
    vcov::Union{Symbol, Expr, Nothing} = :(simple()),
    start::Union{Vector{T}, Nothing} = nothing,
    maxiter::Integer = 10000, limit::Integer = 10,
    trace::Integer = 0,
    convtol::Real = 1.0e-06,
    devtol::Real = 1.0e-08, steptol::Real = 1.0e-08,
    centertol::Real = 1.0e-05, rhotol::Real = 1.0e-04,
    droppc::Bool = true
   ) where T<:Real
```
For an explanation of the options, see the [manual of the alpaca package](https://cran.r-project.org/web/packages/alpaca/index.html).
