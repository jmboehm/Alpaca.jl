

function feglm(df::AbstractDataFrame, m::Model; kwargs...)
    feglm(df, m.f; m.dict..., kwargs...)
end

function feglm(df::AbstractDataFrame, f::Formula;
    fe::Union{Symbol, Expr, Nothing} = nothing,
    vcov::Union{Symbol, Expr, Nothing} = :(simple()),
    weights::Union{Symbol, Expr, Nothing} = nothing,
    subset::Union{Symbol, Expr, Nothing} = nothing,
    maxiter::Integer = 10000, contrasts::Dict = Dict(),
    tol::Real= 1e-8, df_add::Integer = 0,
    save::Union{Bool, Symbol} = false,  method::Symbol = :lsmr, drop_singletons = true
   )
    
    # @rput points
    
    # R"library('fastcluster')"
    # if method == :ward2
    #     R"clusters <- hclust(dist(points), \"ward.D2\")"
    # elseif method == :ward1
    #     R"clusters <- hclust(dist(points), \"ward.D\")"
    # end
    
    # R"clusterCut <- cutree(clusters, $nclusters)"
    # @rget clusterCut
    
    # return clusterCut

    
    R"""
    library(alpaca)
    library(readstata13)
    """
    
    r_df = robject(df)
    r_f = robject(f)

    R"summary($r_df)"
    
    # ctrl <- feglm.control(step.tol = 1e-07, dev.tol = 1e-07,
    #                        pseudo.tol = 1e-07, rho.tol = 1e-04, iter.max = 100L, trace = 2L,
    #                        drop.pc = TRUE)
    R"mod <- feglm(formula = $r_f , data = $r_df, family = binomial())"

    a = @rget mod
    


end
