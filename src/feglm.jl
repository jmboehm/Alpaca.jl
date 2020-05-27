
# todo:
#

# requires the following R packages installed:
#   alpaca
#   formula.tools

#
# function feglm(df::AbstractDataFrame, m::Model; kwargs...)
#     feglm(df, m.f; m.dict..., kwargs...)
# end

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

    # Check fixed effect formula and construct string
    feformula = fe
    fe_str = ""
    has_absorb = feformula != nothing
    if has_absorb
        # check depth 1 symbols in original formula are all CategoricalVector
        if isa(feformula, Symbol)
            x = feformula
            !isa(df[!,x], CategoricalVector) && error("$x should be CategoricalVector")
            fe_str = String(fe)
        elseif feformula.args[1] == :+
            x = feformula.args
            for i in 2:length(x)
                isa(x[i], Symbol) && !isa(df[x[i]], CategoricalVector) && error("$(x[i]) should be CategoricalVector")
                fe_str = "$fe_str + $(String(x[i]))"
            end
        end
    end

    # Start your engines
    R"library(alpaca)"
    R"library(formula.tools)"

    # Check family
    if family == Binomial()
        R"fam <- binomial()"
    elseif family == Normal()
        R"fam <- gaussian()"
    elseif family == Gamma()
        R"fam <- Gamma()"
    elseif family == InverseGaussian()
        R"fam <- inverse.gaussian()"
    elseif family == Poisson()
        R"fam <- poisson()"
    else
        error("Family not supported.")
    end

    if start != nothing
        @rput start
    else
        R"start <- NULL"
    end

    cluster = false
    if vcov == :(simple())
        R"type <- c(\"hessian\")"
        R"str_vcov_formula <- NULL"
    elseif vcov == :robust
        R"type <- c(\"sandwich\")"
        R"str_vcov_formula <- NULL"
    elseif typeof(vcov) == Expr
        (vcov.args[1] == :cluster) || error("Invalid vcov argument.")
        cluster = true
        if isa(vcov.args[2],Symbol)
            clusterstring = string(vcov.args[2])
            fstring = " ~ $(clusterstring)"
            @rput fstring
            @rput clusterstring
            R"vcov_f <- as.formula(fstring)"
            # vcov_f = @formula(String(vcov.args[2]))
            # str_vcov_formula = [String(vcov.args[2])]
        else
            (vcov.args[2].args[1] == :+) || error("Cluster formula specification invalid.")
            clusterstring = string(vcov.args[2])
            fstring = " ~ $(clusterstring)"
            @rput fstring
            @rput clusterstring
            R"vcov_f <- as.formula(fstring)"
            # vcov_f = @formula( ~ String.(vcov.args[2].args[2:end]))
            # str_vcov_formula = String.(vcov.args[2].args[2:end])
        end
        R"type <- c(\"clustered\")"
        # @show str_vcov_formula
        # @rput str_vcov_formula
    else
        error("Invalid vcov argument.")
    end

    @rput trace
    @rput limit

    # move objects to R
    r_df = robject(df)
    r_f = robject(f)
    @rput f
    @rput fe_str

    # Julia's formulae don't have the FE added.
    # Do this, then remove all brackets (alpaca doesn't like them)
    R"fstr <- as.character(update(f, paste(\"~ . |\",fe_str ) ) )"
    if cluster
        R"fstr <- as.character(update(f, paste(\"~ . |\",fe_str, \" |\" ,clusterstring ) ) )"
    end
    R"fstr <- gsub(\"(\", \"\", fstr, fixed=TRUE)"
    R"fstr <- gsub(\")\", \"\", fstr, fixed=TRUE)"
    R"f <- as.formula(fstr)"

    # confirm what we're running
    R"f_string <- as.character(f)"
    f_str = @rget f_string
    println("Running alpaca with formula: $f_str")

    R"ctrl <- feglmControl(dev.tol = $devtol,
       center.tol = $centertol, rho.tol = $rhotol,
       conv.tol = $convtol,
       iter.max = $maxiter, limit = limit,
       trace = trace,
       drop.pc = $droppc)"

    # R"ctrl <- feglm.control(step.tol = $steptol, dev.tol = $devtol,
    #       pseudo.tol = $pseudotol, rho.tol = $rhotol,
    #       iter.max = $maxiter, trace = $trace,
    #       drop.pc = $droppc)"

    R"result <- feglm(formula =  f , data = $r_df,
            family = fam , beta.start = start,
            control = ctrl)"

    if cluster
        output = R"sum <- summary(result, type = type, cluster = vcov_f)"
        R"vcov <- vcov(result, type = type, cluster = vcov_f )"
    else
        output = R"sum <- summary(result, type = type)"
        R"vcov <- vcov(result, type = type )"
    end
    println(output)

    R"coefnames <- names(coef(result))"
    R"yname <- all.vars(result[[\"formula\"]])[1]"
    R"nobs <- result[[\"nobs\"]]"
    R"lvlsk <- result[[\"lvls.k\"]]"
    @rget coefnames
    @rget yname
    @rget nobs
    @rget lvlsk

    println("All done.")
    @rget result
    @rget vcov

    # convert scalars to arrays, if they are scalars
    coef = result[:coefficients]
    coef = isa(coef, Vector) ? coef : [coef]
    # this is a bit hacky...
    vcov = isa(vcov, Array) ? vcov : vcov .+ zeros(Float64,1,1)
    coefnames = isa(coefnames, Vector) ? coefnames : [coefnames]

    # @show lvlsk
    # @show nobs
    # @show coef
    # @show vcov

    rr = RegressionResult(coef,
        vcov,
        coefnames,
        Symbol(yname),
        f,
        nobs[4], sum(lvlsk)
    )

   return rr


end

# Interface with strings. Hacky, and a temporary solution.
# function feglm(df::AbstractDataFrame, formula::String, family::String;
#     vcov::Union{Symbol, Expr, Nothing} = :(simple()))
#
#     # load alpaca in R
#     R"library(alpaca)"
#
#     println("Formula")
#     r_df = robject(df)
#     @rput formula
#     R"f <- as.formula(formula)"
#
#     # run it
#     println("Running feglm")
#     R"result <- feglm(formula =  f , data = $r_df, family = binomial())"
#
#     println("Vcov")
#     if vcov == :(simple())
#         R"type <- c(\"hessian\")"
#         R"str_vcov_formula <- NULL"
#     elseif vcov == :robust
#         R"type <- c(\"sandwich\")"
#         R"str_vcov_formula <- NULL"
#     elseif typeof(vcov) == Expr
#         (vcov.args[1] == :cluster) || error("Invalid vcov argument.")
#         if isa(vcov.args[2],Symbol)
#             str_vcov_formula = [String(vcov.args[2])]
#         else
#             (vcov.args[2].args[1] == :+) || error("Cluster formula specification invalid.")
#             str_vcov_formula = String.(vcov.args[2].args[2:end])
#         end
#         R"type <- c(\"clustered\")"
#         @show str_vcov_formula
#         @rput str_vcov_formula
#     else
#         error("Invalid vcov argument.")
#     end
#
#     println("summary")
#     output = R"sum <- summary(result, type = type,
#            cluster.vars = str_vcov_formula)"
#     println(output)
#
#     R"vcov <- vcov(result, type = type, cluster.vars = str_vcov_formula )"
#     R"coefnames <- names(coef(result))"
#     R"yname <- all.vars(result[[\"formula\"]])[1]"
#     R"nobs <- result[[\"nobs\"]]"
#     R"lvlsk <- result[[\"lvls.k\"]]"
#     @rget coefnames
#     @rget yname
#     @rget nobs
#     @rget lvlsk
#
#     @rget result
#     @rget vcov
#
#     coef = result[:coefficients]
#     coef = isa(coef, Vector) ? coef : [coef]
#     # this is a bit hacky...
#     vcov = isa(vcov, Array) ? vcov : vcov .+ zeros(Float64,1,1)
#     coefnames = isa(coefnames, Vector) ? coefnames : [coefnames]
#
#     rr = RegressionResult(coef,
#         vcov,
#         coefnames,
#         Symbol(yname),
#         @formula(y ~ x),
#         nobs[4], sum(lvlsk)
#     )
#
#    return rr
#
# end

macro vcov(args...)
    Expr(:call, :vcov_helper, (esc(Base.Meta.quot(a)) for a in args)...)
end

function vcov_helper(args...)

    (args[1].head === :call && args[1].args[1] === :(~)) || throw("First argument of @model should be a formula")
    f = @eval(@formula($(args[1].args[2]) ~ $(args[1].args[3])))
    dict = Dict{Symbol, Any}()
    for i in 2:length(args)
        isa(args[i], Expr) &&  args[i].head== :(=) || throw("All arguments of @model, except the first one, should be keyboard arguments")
        dict[args[i].args[1]] = args[i].args[2]
    end
    Model(f, dict)
end
