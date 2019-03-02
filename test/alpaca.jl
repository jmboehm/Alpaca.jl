include("src/alpaca.jl")

using GLM, RDatasets, Test, Distributions

df = dataset("datasets", "iris")
df.binary = vec(Float64.(rand(0:1,size(df,1),1)))
df[:SpeciesDummy] = categorical(df[:Species])
idx = rand(1:3,size(df,1),1)
a = ["A","B","C"]
df.Random = vec([a[i] for i in idx])
df.RandomCategorical = categorical(df[:Random])


m = Alpaca.@model(SepalLength ~ SepalWidth, fe = SpeciesDummy)

rr1 = Alpaca.feglm(df, Alpaca.@model(binary ~ SepalWidth, fe = SpeciesDummy))

# binomial/logit
result = Alpaca.feglm(df, Alpaca.@formula(binary ~ SepalWidth),
    Binomial(), 
    fe = :SpeciesDummy
    )
result = Alpaca.feglm(df, Alpaca.@formula(binary ~ SepalWidth),
    Binomial(), 
    fe = :SpeciesDummy, vcov = :(cluster(SpeciesDummy))
    )
result = Alpaca.feglm(df, Alpaca.@formula(binary ~ SepalWidth),
    Binomial(), 
    fe = :SpeciesDummy, vcov = :(cluster(SpeciesDummy + RandomCategorical))
    );

# poisson
rr1 = Alpaca.feglm(df, Alpaca.@formula(binary ~ SepalWidth),
    Poisson(), 
    fe = :SpeciesDummy
    )


rr1 = Alpaca.feglm(df, "binary ~ SepalWidth | SpeciesDummy", "Binomial")
 


using FixedEffectModels
reg(df, @model(SepalLength ~ SepalWidth, fe = SpeciesDummy, vcov = cluster(SpeciesDummy)))

m = @model(SepalLength ~ SepalWidth, fe = SpeciesDummy, vcov = cluster(SpeciesDummy))

