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
# testing display of RegressionResult
@show result
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


rng = MersenneTwister(1234)
N = 1_000_000
K = 100
id1 = rand(rng, 1:(round(Int64,N/K)), N)
id2 = rand(rng, 1:K, N)
x1 =  randn(rng, N) ./ 10.0
x2 =  randn(rng, N) ./ 10.0
y= exp.(3.0 .* x1 .+ 2.0 .* x2 .+ sin.(id1) .+ cos.(id2).^2 .+ randn(rng, N))
df = DataFrame(id1 = categorical(id1), id2 = categorical(id2), x1 = x1, x2 = x2, y = y)
df = DataFrame(id1_noncat = id1, id1 = categorical(id1), id2 = categorical(id2), x1 = x1, x2 = x2, y = y)

f = Alpaca.@formula(y ~ x1 + x2)
result = Alpaca.feglm(df, Alpaca.@formula(y ~ x1 + x2),
    Poisson(),
    fe = :id2,
    start = [0.2;0.2], trace = 2
    )






# string interface (not essential)
# rr1 = Alpaca.feglm(df, "binary ~ SepalWidth | SpeciesDummy", "Binomial")
# @test StatsBase.coef(rr1) ≈ [-0.221486] atol = 1e-4
# rr1 = Alpaca.feglm(df, "binary ~ SepalWidth | SpeciesDummy", "Binomial", vcov = :(cluster(SpeciesDummy)))
# @test StatsBase.stderror(rr1) ≈ [0.317681] atol = 1e-4
