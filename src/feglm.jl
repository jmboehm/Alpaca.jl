
# Ideally, use a similar interface as GLM.jl or FixedEffectModels.jl
# But now we're lazy and use strings instead...

# function feglm(df::AbstractDataFrame, m::Model; kwargs...)
#     feglm(df, m.f; m.dict..., kwargs...)
# end

# function feglm(df::AbstractDataFrame, f::Formula;
#     fe::Union{Symbol, Expr, Nothing} = nothing,
#     vcov::Union{Symbol, Expr, Nothing} = :(simple()),
#     weights::Union{Symbol, Expr, Nothing} = nothing,
#     subset::Union{Symbol, Expr, Nothing} = nothing,
#     maxiter::Integer = 10000, contrasts::Dict = Dict(),
#     tol::Real= 1e-8, df_add::Integer = 0,
#     save::Union{Bool, Symbol} = false,  method::Symbol = :lsmr, drop_singletons = true
#    )

#     # Check fixed effect formula and construct string
#     feformula = fe
#     fe_str = "" 
#     has_absorb = feformula != nothing
#     if has_absorb
#         # check depth 1 symbols in original formula are all CategoricalVector
#         if isa(feformula, Symbol)
#             x = feformula
#             !isa(df[x], CategoricalVector) && error("$x should be CategoricalVector")
#             fe_str = String(fe)
#         elseif feformula.args[1] == :+
#             x = feformula.args
#             for i in 2:length(x)
#                 isa(x[i], Symbol) && !isa(df[x[i]], CategoricalVector) && error("$(x[i]) should be CategoricalVector")
#                 fe_str = "$fe_str + $(String(x[i]))"
#             end
#         end
#     end
    
#     R"library(alpaca)"
    
#     println("FE String: $fe_str")

#     r_df = robject(df)
#     r_f = robject(f)
#     @rput f
#     @rput fe_str
    
#     # Julia's formulae don't have the FE added. 
#     # For some reason, this does not work (brackets screw it up)
#     #R"f <- update(f, paste(\"~ . |\",fe_str ) )"

#     # for some reason, this works though:
#     #R"f <- as.formula(binary ~ SepalWidth | SpeciesDummy)"

#     # confirm what we're running
#     R"f_string <- as.character(f)"
#     f_str = @rget f_string
#     println("Running alpaca with formula: $f_str")

#     # ctrl <- feglm.control(step.tol = 1e-07, dev.tol = 1e-07,
#     #                        pseudo.tol = 1e-07, rho.tol = 1e-04, iter.max = 100L, trace = 2L,
#     #                        drop.pc = TRUE)
#     R"mod <- feglm(formula =  f , data = $r_df, family = binomial())"

#     @rget mod
# end

# Interface with strings. Hacky, and a temporary solution.
function feglm(df::AbstractDataFrame, ms::String)

    # load alpaca in R
    R"library(alpaca)"

    r_df = robject(df)
    @rput ms
    R"f <- as.formula(ms)"

    # run it
    R"mod <- feglm(formula =  f , data = $r_df, family = binomial())"

    # return output
    a = @rget mod

end