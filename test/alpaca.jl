include("src/alpaca.jl")

using GLM, RDatasets, Test

df = dataset("datasets", "iris")

m = Alpaca.@model(SepalLength ~ SepalWidth)

rr1 = Alpaca.feglm(df, Alpaca.@model(SepalLength ~ SepalWidth))
