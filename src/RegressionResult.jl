
struct RegressionResult <: RegressionModel
    coef::Vector{Float64}   # Vector of coefficients
    vcov::Matrix{Float64}   # Covariance matrix

    coefnames::Vector       # Name of coefficients
    yname::Symbol           # Name of dependent variable
    formula::FormulaTerm        # Original formula

    nobs::Int64             # Number of observations
    dof_residual::Int64      # degrees of freedoms

    # rss::Float64            # Sum of squared residuals
    # tss::Float64            # Total sum of squares
    # r2::Float64             # R squared
    # adjr2::Float64          # R squared adjusted
    # F::Float64              # F statistics
    # p::Float64              # p value for the F statistics
end

StatsBase.nobs(x::RegressionResult) = x.nobs
StatsBase.coef(x::RegressionResult) = x.coef
StatsBase.coefnames(x::RegressionResult) = x.coefnames
StatsBase.vcov(x::RegressionResult) = x.vcov
StatsBase.dof_residual(x::RegressionResult) = x.dof_residual

function StatsBase.confint(x::RegressionResult)
    scale = quantile(TDist(x.dof_residual), 1 - (1-0.95)/2)
    se = stderror(x)
    hcat(x.coef -  scale * se, x.coef + scale * se)
end

# Display Results
function title(::RegressionResult)
    "Generalized Linear Fixed Effects Model"
end

function top(::RegressionResult)
    error("function top has no general method for AbstractRegressionResult")
end

top(x::RegressionResult) = [
            "Number of obs" sprint(show, nobs(x), context = :compact => true);
            "Degrees of freedom" sprint(show, nobs(x) - dof_residual(x), context = :compact => true);
            ]

function coeftable(x::RegressionResult)
    ctitle = title(x)
    ctop = top(x)
    cc = coef(x)
    se = stderror(x)
    coefnms = coefnames(x)
    conf_int = confint(x)
    # put (intercept) last
    if !isempty(coefnms) && ((coefnms[1] == Symbol("(Intercept)")) || (coefnms[1] == "(Intercept)"))
        newindex = vcat(2:length(cc), 1)
        cc = cc[newindex]
        se = se[newindex]
        conf_int = conf_int[newindex, :]
        coefnms = coefnms[newindex]
    end
    tt = cc ./ se
    CoefTable2(
        hcat(cc, se, tt, ccdf.(Ref(FDist(1, dof_residual(x))), abs2.(tt)), conf_int[:, 1:2]),
        ["Estimate","Std.Error","t value", "Pr(>|t|)", "Lower 95%", "Upper 95%" ],
        ["$(coefnms[i])" for i = 1:length(cc)], 4, ctitle, ctop)
end

function Base.show(io::IO, x::RegressionResult)
    show(io, coeftable(x))
end


## Coeftalble2 is a modified Coeftable allowing for a top String matrix displayed before the coefficients.
## Pull request: https://github.com/JuliaStats/StatsBase.jl/pull/119

struct CoefTable2
    mat::Matrix
    colnms::Vector
    rownms::Vector
    pvalcol::Integer
    title::AbstractString
    top::Matrix{AbstractString}
    function CoefTable2(mat::Matrix,colnms::Vector,rownms::Vector,pvalcol::Int=0,
                        title::AbstractString = "", top::Matrix = Any[])
        nr,nc = size(mat)
        0 <= pvalcol <= nc || error("pvalcol = $pvalcol should be in 0,...,$nc]")
        length(colnms) in [0,nc] || error("colnms should have length 0 or $nc")
        length(rownms) in [0,nr] || error("rownms should have length 0 or $nr")
        length(top) == 0 || (size(top, 2) == 2 || error("top should have 2 columns"))
        new(mat,colnms,rownms,pvalcol, title, top)
    end
end


## format numbers in the p-value column
function format_scientific(pv::Number)
    return @sprintf("%.3f", pv)
end


function Base.show(io::IO, ct::CoefTable2)
    mat = ct.mat; nr,nc = size(mat); rownms = ct.rownms; colnms = ct.colnms;
    pvc = ct.pvalcol; title = ct.title;   top = ct.top
    if length(rownms) == 0
        rownms = AbstractString[lpad("[$i]",floor(Integer, log10(nr))+3) for i in 1:nr]
    end
    if length(rownms) > 0
        rnwidth = max(4,maximum([length(nm) for nm in rownms]) + 1)
        else
            # if only intercept, rownms is empty collection, so previous would return error
        rnwidth = 4
    end
    rownms = [rpad(nm,rnwidth) for nm in rownms]
    widths = [length(cn)::Int for cn in colnms]
    str = [sprint(show, mat[i,j]; context=:compact => true) for i in 1:nr, j in 1:nc]
    if pvc != 0                         # format the p-values column
        for i in 1:nr
            str[i, pvc] = format_scientific(mat[i, pvc])
        end
    end
    for j in 1:nc
        for i in 1:nr
            lij = length(str[i, j])
            if lij > widths[j]
                widths[j] = lij
            end
        end
    end
    widths .+= 1
    totalwidth = sum(widths) + rnwidth
    if length(title) > 0
        halfwidth = div(totalwidth - length(title), 2)
        println(io, " " ^ halfwidth * string(title) * " " ^ halfwidth)
    end
    if length(top) > 0
        for i in 1:size(top, 1)
            top[i, 1] = top[i, 1] * ":"
        end
        println(io, "=" ^totalwidth)
        halfwidth = div(totalwidth, 2) - 1
        interwidth = 2 +  mod(totalwidth, 2)
        for i in 1:(div(size(top, 1) - 1, 2)+1)
            print(io, top[2*i-1, 1])
            print(io, lpad(top[2*i-1, 2], halfwidth - length(top[2*i-1, 1])))
            print(io, " " ^interwidth)
            if size(top, 1) >= 2*i
                print(io, top[2*i, 1])
                print(io, lpad(top[2*i, 2], halfwidth - length(top[2*i, 1])))
            end
            println(io)
        end
    end
    println(io,"=" ^totalwidth)
    println(io," " ^ rnwidth *
            join([lpad(string(colnms[i]), widths[i]) for i = 1:nc], ""))
    println(io,"-" ^totalwidth)
    for i in 1:nr
        print(io, rownms[i])
        for j in 1:nc
            print(io, lpad(str[i,j],widths[j]))
        end
        println(io)
    end
    println(io,"=" ^totalwidth)
end
