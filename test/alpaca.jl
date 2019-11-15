using Alpaca, RDatasets, Test, Distributions
using Random

rng = MersenneTwister(1234)

df = dataset("datasets", "iris")
df.binary = vec(Float64.(rand(rng,0:1,size(df,1),1)))
df[!,:SpeciesDummy] = categorical(df[!,:Species])
idx = rand(rng,1:3,size(df,1),1)
a = ["A","B","C"]
df.Random = vec([a[i] for i in idx])
df.RandomCategorical = categorical(df[!,:Random])

# proper interface

# try it natively in R
# @rput df
# R"mod <- feglm(binary ~ SepalWidth | SpeciesDummy , df, family = binomial())"
# R"summary(mod, \"hessian\")"
# binomial/logit
result = Alpaca.feglm(df, Alpaca.@formula(binary ~ SepalWidth),
    Binomial(),
    fe = :SpeciesDummy
    )
@test StatsBase.coef(result) ≈ [-0.221486] atol = 1e-4
# with clustering
# R"mod <- feglm(binary ~ SepalWidth | SpeciesDummy | SpeciesDummy, df, family = binomial())"
# R"summary(mod, \"cluster\", cluster = ~ SpeciesDummy)"
result = Alpaca.feglm(df, Alpaca.@formula(binary ~ SepalWidth),
    Binomial(),
    fe = :SpeciesDummy, vcov = :(cluster(SpeciesDummy))
    )
@test StatsBase.stderror(result) ≈ [0.317681] atol = 1e-4
# two-way clustering
# R"mod <- feglm(binary ~ SepalWidth | SpeciesDummy | SpeciesDummy + RandomCategorical, df, family = binomial())"
# R"summary(mod, \"cluster\", cluster = ~ SpeciesDummy + RandomCategorical)"
result = Alpaca.feglm(df, Alpaca.@formula(binary ~ SepalWidth),
    Binomial(),
    fe = :SpeciesDummy, vcov = :(cluster(SpeciesDummy + RandomCategorical))
    )
@test StatsBase.stderror(result) ≈ [0.228229] atol = 1e-4
# poisson
# R"mod <- feglm(binary ~ SepalWidth | SpeciesDummy, df, family = poisson())"
# R"summary(mod, \"hessian\")"
result = Alpaca.feglm(df, Alpaca.@formula(binary ~ SepalWidth),
    Poisson(),
    fe = :SpeciesDummy
    )
@test StatsBase.coef(result) ≈ [-0.103008] atol = 1e-4
# two fe's
# R"mod <- feglm(binary ~ SepalWidth | SpeciesDummy + RandomCategorical, df, family = poisson())"
# R"summary(mod, \"hessian\")"
result = Alpaca.feglm(df, Alpaca.@formula(binary ~ SepalWidth),
    Poisson(),
    fe =:(SpeciesDummy + RandomCategorical)
    )
@test StatsBase.coef(result) ≈ [-0.153311] atol = 1e-4


# string interface (not essential)
# rr1 = Alpaca.feglm(df, "binary ~ SepalWidth | SpeciesDummy", "Binomial")
# @test StatsBase.coef(rr1) ≈ [-0.221486] atol = 1e-4
# rr1 = Alpaca.feglm(df, "binary ~ SepalWidth | SpeciesDummy", "Binomial", vcov = :(cluster(SpeciesDummy)))
# @test StatsBase.stderror(rr1) ≈ [0.317681] atol = 1e-4
