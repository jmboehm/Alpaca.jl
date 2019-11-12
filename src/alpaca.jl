module Alpaca

    ##############################################################################
    ##
    ## Dependencies
    ##
    ##############################################################################

    using Base
    using Statistics
    using RCall
    using JLD2
    using DataFrames
    using Distributions
    using Printf
    # using StatsBase: CoefTable, StatisticalModel, RegressionModel
    # import Printf: @sprintf
    # import DataFrames: DataFrame, AbstractDataFrame, completecases, names!, ismissing
    import StatsBase: coef, nobs, coeftable, vcov, predict, residuals, var,
        RegressionModel, model_response, stderror, confint, fit, CoefTable,
        dof_residual, r2, adjr2, deviance, mss, rss, islinear, response
    # import StatsModels: @formula,  FormulaTerm, ModelFrame, ModelMatrix, coefnames
    using Reexport
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

    include("model.jl")
    include("RegressionResult.jl")
    include("feglm.jl")


end
