module Alpaca

    ##############################################################################
    ##
    ## Dependencies
    ##
    ##############################################################################

    using Base
    using Statistics
    using RCall
    using DataFrames
    using Distributions
    using Printf
    import StatsBase: coef, nobs, coeftable, vcov, predict, residuals, var,
        RegressionModel, model_response, stderror, confint, fit, CoefTable,
        dof_residual, r2, adjr2, deviance, mss, rss, islinear, response
    using Reexport
    using CategoricalArrays
    @reexport using StatsBase
    @reexport using StatsModels

    ##############################################################################
    ##
    ## Exported methods and types
    ##
    ##############################################################################

    export feglm

    ##############################################################################
    ##
    ## Load files
    ##
    ##############################################################################

    include("RegressionResult.jl")
    include("feglm.jl")

    # instantiate R's alpaca package
    # R"if(\"alpaca\" %in% rownames(installed.packages()) == FALSE) {install.packages(\"alpaca\")}"
    # R"if(\"formula.tools\" %in% rownames(installed.packages()) == FALSE) {install.packages(\"formula.tools\")}"

end
