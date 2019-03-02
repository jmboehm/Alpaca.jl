module Alpaca

    ##############################################################################
    ##
    ## Dependencies
    ##
    ##############################################################################
    
    using RCall
    using JLD2
    using DataFrames
    using Distributions
    using StatsBase: CoefTable, StatisticalModel, RegressionModel
    import Printf: @sprintf
    import DataFrames: DataFrame, AbstractDataFrame, completecases, names!, ismissing
    import StatsBase: coef, nobs, coeftable, vcov, predict, residuals, var, RegressionModel, model_response, stderror, confint, fit, CoefTable, dof_residual,  df_residual, r2, adjr2, deviance, mss, rss, islinear, response
    import StatsModels: @formula,  Formula, ModelFrame, ModelMatrix, Terms, coefnames, evalcontrasts, check_non_redundancy!
    using Reexport
    @reexport using StatsBase
    @reexport using StatsModels

    ##############################################################################
    ##
    ## Exported methods and types
    ##
    ##############################################################################

    export feglm,
    Model,
    @model

    ##############################################################################
    ##
    ## Load files
    ##
    ##############################################################################
    
    include("model.jl")
    include("RegressionResult.jl")
    include("feglm.jl")


end