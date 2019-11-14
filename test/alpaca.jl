using Alpaca, RDatasets, Test, Distributions

df = dataset("datasets", "iris")
df.binary = vec(Float64.(rand(0:1,size(df,1),1)))
df[!,:SpeciesDummy] = categorical(df[!,:Species])
idx = rand(1:3,size(df,1),1)
a = ["A","B","C"]
df.Random = vec([a[i] for i in idx])
df.RandomCategorical = categorical(df[!,:Random])

# proper interface

# binomial/logit
result = Alpaca.feglm(df, Alpaca.@formula(binary ~ SepalWidth),
    Binomial(),
    fe = :SpeciesDummy
    )
@test StatsBase.coef(result) ≈ [-1.00555] atol = 1e-4
# with clustering
result = Alpaca.feglm(df, Alpaca.@formula(binary ~ SepalWidth),
    Binomial(),
    fe = :SpeciesDummy, vcov = :(cluster(SpeciesDummy))
    )
@test StatsBase.stderror(result) ≈ [0.464767] atol = 1e-4
# two-way clustering
result = Alpaca.feglm(df, Alpaca.@formula(binary ~ SepalWidth),
    Binomial(),
    fe = :SpeciesDummy, vcov = :(cluster(SpeciesDummy + RandomCategorical))
    )
@test StatsBase.stderror(result) ≈ [0.446949] atol = 1e-4
# poisson
result = Alpaca.feglm(df, Alpaca.@formula(binary ~ SepalWidth),
    Poisson(),
    fe = :SpeciesDummy
    )
@test StatsBase.coef(result) ≈ [-0.507638] atol = 1e-4
# two fe's
result = Alpaca.feglm(df, Alpaca.@formula(binary ~ SepalWidth),
    Poisson(),
    fe =:(SpeciesDummy + RandomCategorical)
    )
@test StatsBase.coef(result) ≈ [-0.502443] atol = 1e-4


# string interface
rr1 = Alpaca.feglm(df, "binary ~ SepalWidth | SpeciesDummy", "Binomial")
@test StatsBase.coef(rr1) ≈ [-1.00555] atol = 1e-4
rr1 = Alpaca.feglm(df, "binary ~ SepalWidth | SpeciesDummy", "Binomial", vcov = :(cluster(SpeciesDummy)))
@test StatsBase.stderror(rr1) ≈ [0.464767] atol = 1e-4
