include("src/alpaca.jl")

using GLM, RDatasets, Test

df = dataset("datasets", "iris")
df.binary = vec(Float64.(rand(0:1,size(df,1),1)))
df[:SpeciesDummy] = categorical(df[:Species])

m = Alpaca.@model(SepalLength ~ SepalWidth, fe = SpeciesDummy)

rr1 = Alpaca.feglm(df, Alpaca.@model(binary ~ SepalWidth, fe = SpeciesDummy))

rr1 = Alpaca.feglm(df, "binary ~ SepalWidth | SpeciesDummy")
